import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { clearSpeechPrefetch, playSpeechText, prefetchSpeechText } from './voice-playback'

const speakText = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', () => ({ speakText }))

const options = { source: 'voice-conversation' as const }

// jsdom has no media stack: an <audio> never loads and play() is unimplemented.
// This stands in for the element so the decode/play path can be driven, firing
// `canplaythrough` when a src is set and `ended` once play() is called.
class FakeAudio extends EventTarget {
  preload = ''
  #src = ''

  set src(value: string) {
    this.#src = value
    queueMicrotask(() => this.dispatchEvent(new Event('canplaythrough')))
  }

  get src() {
    return this.#src
  }

  load = vi.fn()
  pause = vi.fn()
  play = vi.fn(async () => {
    queueMicrotask(() => this.dispatchEvent(new Event('ended')))
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  clearSpeechPrefetch()
  vi.stubGlobal('Audio', FakeAudio)
  speakText.mockResolvedValue({ data_url: 'data:audio/wav;base64,AAAA' })
  $voicePlayback.set({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
})

afterEach(() => {
  vi.unstubAllGlobals()
  clearSpeechPrefetch()
})

// Settle the prefetch chain (speakText -> canplaythrough) before playing.
const settle = () => new Promise(resolve => setTimeout(resolve, 0))

describe('speech prefetch', () => {
  // The point of the cache: the sentence the loop prefetched while the previous
  // one played is already synthesized and decoded, so playing it must not
  // synthesize a second time.
  it('plays a prefetched sentence without synthesizing again', async () => {
    prefetchSpeechText('Hello there friend.')
    await settle()
    expect(speakText).toHaveBeenCalledTimes(1)

    await expect(playSpeechText('Hello there friend.', options)).resolves.toBe(true)

    expect(speakText).toHaveBeenCalledTimes(1)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('synthesizes on demand when the sentence was not prefetched', async () => {
    await expect(playSpeechText('Nothing cached here.', options)).resolves.toBe(true)

    expect(speakText).toHaveBeenCalledTimes(1)
  })

  // A prefetch whose synth failed must not poison the sentence: the entry is
  // consumed, the play path re-synthesizes, and that attempt decides the result.
  it('re-synthesizes when the prefetched entry failed, and plays', async () => {
    speakText.mockRejectedValueOnce(new Error('transient 503'))
    prefetchSpeechText('Recover from this.')
    await settle()

    await expect(playSpeechText('Recover from this.', options)).resolves.toBe(true)

    expect(speakText).toHaveBeenCalledTimes(2)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('reports the error when the prefetch failed and the retry fails too', async () => {
    speakText.mockRejectedValue(new Error('backend down'))
    prefetchSpeechText('No luck here.')
    await settle()

    await expect(playSpeechText('No luck here.', options)).rejects.toThrow('backend down')
    expect($voicePlayback.get().status).toBe('idle')
  })

  // An unconsumed prefetch that failed must stay quiet: its rejection is absorbed
  // where it is cached, not left to surface as an unhandled rejection.
  it('does not raise an unhandled rejection for a prefetch nobody plays', async () => {
    const unhandled = vi.fn()
    process.on('unhandledRejection', unhandled)
    speakText.mockRejectedValue(new Error('never consumed'))

    prefetchSpeechText('Dropped sentence.')
    await settle()
    await settle()

    process.off('unhandledRejection', unhandled)
    expect(unhandled).not.toHaveBeenCalled()
  })
})

describe('playSpeechText synthesis failure', () => {
  // 'preparing' reads as active to `isVoicePlaybackActive`, which gates
  // auto-speak, Read Aloud and the mic re-arm. Wedging it there disables voice
  // mode until reload, so a failed synth has to leave the store idle.
  it('resets the playback state to idle when synthesis fails', async () => {
    speakText.mockRejectedValue(new Error('backend down'))

    await expect(playSpeechText('Hello there.', options)).rejects.toThrow('backend down')

    expect($voicePlayback.get().status).toBe('idle')
  })

  it('propagates the synthesis error to the caller', async () => {
    speakText.mockRejectedValue(new Error('backend down'))

    await expect(playSpeechText('Hello there.', options)).rejects.toThrow('backend down')
  })
})
