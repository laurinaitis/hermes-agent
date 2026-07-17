import { describe, expect, it } from 'vitest'

import { extractChunk } from './use-voice-conversation'

const take = (text: string, force = false) => extractChunk(text, force)

describe('extractChunk', () => {
  it('splits on a sentence boundary once the chunk is long enough to voice', () => {
    expect(take('This is a full sentence. And another one.')).toEqual([
      'This is a full sentence.',
      'And another one.'
    ])
  })

  it('keeps a decimal whole', () => {
    expect(take('Pi is 2.0 and that is fine. Next sentence here.')).toEqual([
      'Pi is 2.0 and that is fine.',
      'Next sentence here.'
    ])
  })

  // The reason MIN_SPEAK_CHARS accumulates: a bare list marker is too short to
  // voice on its own, so it rides along with the item it introduces.
  it('keeps a list marker attached to its item', () => {
    expect(take('1. First take the lid off. 2. Then pour.')).toEqual([
      '1. First take the lid off.',
      '2. Then pour.'
    ])
  })

  // A sentence ending in a number must still end the sentence. Mid-stream this
  // is the difference between speaking now and speaking nothing until more text
  // arrives, which matters most for exactly the numeric replies voice mode is
  // slowest on.
  it('ends a sentence that finishes on a number', () => {
    expect(take('It closed at 187.5. Volume was up today.')).toEqual([
      'It closed at 187.5.',
      'Volume was up today.'
    ])
    expect(take('The answer is 42.')).toEqual(['The answer is 42.', ''])
  })

  // The accepted cost of treating "42." and "2." alike: they are the same token,
  // so a list of very short items glues the next marker onto the previous chunk.
  // Audible only as an early boundary, and preferred over stalling on numbers.
  it('glues a trailing marker when every list item is shorter than the minimum', () => {
    expect(take('1. Yes. 2. No.')).toEqual(['1. Yes. 2.', 'No.'])
  })

  it('waits rather than voicing a fragment with no boundary', () => {
    expect(take('a partial sentence with no end yet')).toBeNull()
  })

  it('voices what is left when forced', () => {
    expect(take('a partial sentence with no end yet', true)).toEqual([
      'a partial sentence with no end yet',
      ''
    ])
  })

  it('soft-wraps a long boundary-free run at a clause break', () => {
    const runOn = `${'x'.repeat(120)}, ${'y'.repeat(120)}`
    const result = take(runOn)

    expect(result).not.toBeNull()
    expect(result?.[0].length).toBeLessThan(runOn.length)
    expect(`${result?.[0]} ${result?.[1]}`).toBe(runOn)
  })

  // `。！？` end a sentence the same way `.` does. Note the boundary still needs
  // whitespace or the buffer end after it, so unspaced CJK runs stay whole -
  // same as before this splitter, and MIN_SPEAK_CHARS counts characters, so
  // short CJK sentences merge.
  it('ends a sentence on CJK punctuation', () => {
    expect(take('これは長い日本語の文です。 次の文もあります。')).toEqual([
      'これは長い日本語の文です。',
      '次の文もあります。'
    ])
  })
})
