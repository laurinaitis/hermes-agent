import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { clearSpeechPrefetch, playSpeechText } from './voice-playback'

const speakText = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', () => ({ speakText }))

const options = { source: 'voice-conversation' as const }

beforeEach(() => {
  vi.clearAllMocks()
  clearSpeechPrefetch()
  $voicePlayback.set({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
})

describe('playSpeechText synthesis failure', () => {
  // A failed synth has to leave the store idle: 'preparing' reads as active to
  // `isVoicePlaybackActive`, which gates auto-speak, Read Aloud and (with
  // #55581) the mic re-arm. Wedging it there disables voice mode until reload.
  it('resets the playback state to idle when synthesis fails', async () => {
    speakText.mockRejectedValue(new Error('backend down'))

    await expect(playSpeechText('Hello there.', options)).rejects.toThrow('backend down')

    expect($voicePlayback.get().status).toBe('idle')
  })

  // The caller notifies off the rejection; swallowing it fails silently.
  it('propagates the synthesis error to the caller', async () => {
    speakText.mockRejectedValue(new Error('backend down'))

    await expect(playSpeechText('Hello there.', options)).rejects.toThrow('backend down')
  })

  // A prefetch that failed must not poison the sentence: the cached rejection is
  // consumed, and the on-demand retry decides the outcome.
  it('surfaces the failure when the prefetched entry failed to synthesize', async () => {
    speakText.mockRejectedValue(new Error('backend down'))

    await expect(playSpeechText('Hello there.', options)).rejects.toThrow()
    expect($voicePlayback.get().status).toBe('idle')
  })
})
