import { speakText } from '@/hermes'
import {
  $voicePlayback,
  setVoicePlaybackState,
  type VoicePlaybackSource,
  type VoicePlaybackState
} from '@/store/voice-playback'

import { sanitizeTextForSpeech } from './speech-text'

// Free Edge TTS occasionally hands back audio that never fires `playing`/`ended`
// nor `error` — leaving voice mode stuck "speaking" forever. Reject if playback
// fails to start or stalls mid-stream for this long (rearmed on each progress
// tick, so legitimately long speech is never cut off).
const PLAYBACK_STALL_MS = 15_000

let currentAudio: HTMLAudioElement | null = null
let currentStop: (() => void) | null = null
let sequence = 0

function currentState(
  status: VoicePlaybackState['status'],
  options?: VoicePlaybackOptions,
  audioElement: HTMLAudioElement | null = null
): VoicePlaybackState {
  return {
    audioElement,
    messageId: options?.messageId ?? null,
    sequence,
    source: options?.source ?? null,
    status
  }
}

export interface VoicePlaybackOptions {
  messageId?: string | null
  source: VoicePlaybackSource
}

export function stopVoicePlayback() {
  sequence += 1
  currentStop?.()
  currentStop = null

  if (currentAudio) {
    currentAudio.pause()
    currentAudio.src = ''
    currentAudio.load()
    currentAudio = null
  }

  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence,
    source: null,
    status: 'idle'
  })
}

// Look-ahead playback cache. The voice-conversation loop calls
// `prefetchSpeechText` for the next sentences while the current one is still
// playing. We cache a fully *decoded* HTMLAudioElement (not just the synthesized
// data URL) so the next chunk starts with no "preparing"/decode delay — the
// element is buffered to `canplaythrough` ahead of time. Keyed by the sanitized
// text (the same key `playSpeechText` looks up). Best-effort: a miss falls back
// to synth-on-demand (prior behavior).
const speechPrefetch = new Map<string, Promise<HTMLAudioElement | null>>()

// Synthesize a sentence and buffer/decode it into a ready-to-play element.
// Resolves once the browser reports it can play through (or on a short timeout /
// error — `play()` will still attempt it). Returns null only if synthesis fails.
function prepareSpeechAudio(speakableText: string): Promise<HTMLAudioElement | null> {
  return speakText(speakableText)
    .then(
      response =>
        new Promise<HTMLAudioElement | null>(resolve => {
          const audio = new Audio()
          audio.preload = 'auto'

          let settled = false
          const finish = () => {
            if (settled) {
              return
            }

            settled = true
            audio.removeEventListener('canplaythrough', finish)
            audio.removeEventListener('error', finish)
            resolve(audio)
          }

          audio.addEventListener('canplaythrough', finish, { once: true })
          audio.addEventListener('error', finish, { once: true })
          audio.src = response.data_url
          audio.load()
          // Data URLs usually decode fast; don't block forever if the event is
          // flaky for a given codec.
          window.setTimeout(finish, 600)
        })
    )
    .catch(() => null)
}

export function prefetchSpeechText(text: string): void {
  const speakable = sanitizeTextForSpeech(text)

  if (!speakable || speechPrefetch.has(speakable)) {
    return
  }

  speechPrefetch.set(speakable, prepareSpeechAudio(speakable))

  // Bound the cache so abandoned prefetches (e.g. chunk boundaries that shifted
  // as the stream grew) can't accumulate.
  if (speechPrefetch.size > 6) {
    const oldest = speechPrefetch.keys().next().value

    if (oldest !== undefined) {
      speechPrefetch.delete(oldest)
    }
  }
}

export function clearSpeechPrefetch(): void {
  speechPrefetch.clear()
}

export async function playSpeechText(text: string, options: VoicePlaybackOptions): Promise<boolean> {
  stopVoicePlayback()

  const speakableText = sanitizeTextForSpeech(text)

  if (!speakableText) {
    return false
  }

  const ownSequence = sequence
  const isCurrent = () => ownSequence === sequence

  setVoicePlaybackState(currentState('preparing', options))

  try {
    // Use a preloaded, already-decoded element when the loop prefetched this
    // sentence; otherwise synthesize + buffer it now.
    let audio: HTMLAudioElement | null = null
    const prefetched = speechPrefetch.get(speakableText)

    if (prefetched) {
      speechPrefetch.delete(speakableText)
      audio = await prefetched
    }

    if (!audio) {
      audio = await prepareSpeechAudio(speakableText)
    }

    if (!audio) {
      return false
    }

    if (!isCurrent()) {
      return false
    }

    currentAudio = audio
    setVoicePlaybackState(currentState('speaking', options, audio))

    await new Promise<void>((resolve, reject) => {
      let stall: number | null = null

      const cleanup = () => {
        if (stall !== null) {
          window.clearTimeout(stall)
          stall = null
        }

        audio.removeEventListener('ended', onEnded)
        audio.removeEventListener('error', onError)
        audio.removeEventListener('timeupdate', armStall)
        currentStop = null
      }

      const armStall = () => {
        if (stall !== null) {
          window.clearTimeout(stall)
        }

        stall = window.setTimeout(() => {
          cleanup()
          reject(new Error('Playback stalled'))
        }, PLAYBACK_STALL_MS)
      }

      const onEnded = () => {
        cleanup()
        resolve()
      }

      const onError = () => {
        cleanup()
        reject(new Error('Playback failed'))
      }

      currentStop = () => {
        cleanup()
        resolve()
      }

      audio.addEventListener('ended', onEnded, { once: true })
      audio.addEventListener('error', onError, { once: true })
      audio.addEventListener('timeupdate', armStall)
      armStall()
      void audio.play().catch(onError)
    })

    if (!isCurrent()) {
      return false
    }

    currentAudio = null
    setVoicePlaybackState(currentState('idle'))

    return true
  } catch (error) {
    if (isCurrent()) {
      currentStop = null
      currentAudio = null
      setVoicePlaybackState(currentState('idle'))
    }

    throw error
  }
}

export function isVoicePlaybackActive() {
  return $voicePlayback.get().status !== 'idle'
}
