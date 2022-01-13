// RUN: not llvm-mc -triple aarch64-linux-gnu %s 2> %t > /dev/null
// RUN: FileCheck %s < %t

    .cpu invalid
    // CHECK: error: unknown CPU name

    .cpu generic+wibble+nowobble
    // CHECK: :[[@LINE-1]]:18: error: unsupported architectural extension
    // CHECK: :[[@LINE-2]]:25: error: unsupported architectural extension

    .cpu generic+nofp
    fminnm d0, d0, d1
    // CHECK: error: instruction requires: fp-armv8
    // CHECK-NEXT:   fminnm d0, d0, d1
    // CHECK-NEXT:   ^

    .cpu generic+nosimd
    addp v0.4s, v0.4s, v0.4s
    // CHECK: error: instruction requires: neon
    // CHECK-NEXT:   addp v0.4s, v0.4s, v0.4s
    // CHECK-NEXT:   ^

    .cpu generic+nocrc
    crc32cx w0, w1, x3
    // CHECK: error: instruction requires: crc
    // CHECK-NEXT:   crc32cx w0, w1, x3
    // CHECK-NEXT:   ^

    .cpu generic+nocrypto+crc
    aesd v0.16b, v2.16b
    // CHECK: error: instruction requires: aes
    // CHECK-NEXT:   aesd v0.16b, v2.16b
    // CHECK-NEXT:   ^

    .cpu generic+nolse
    casa  w5, w7, [x20]
    // CHECK: error: instruction requires: lse
    // CHECK-NEXT:   casa  w5, w7, [x20]
    // CHECK-NEXT:   ^

    .cpu generic+v8.1-a
    // CHECK: error: unsupported architectural extension
    // CHECK-NEXT:   .cpu generic+v8.1-a
    // CHECK-NEXT:   ^

    .cpu generic+noaes
    aese v0.16b, v1.16b
    // CHECK:       error: instruction requires: aes
    // CHECK-NEXT:  aese v0.16b, v1.16b
    // CHECK-NEXT:  ^

    .cpu generic+nosha2
    sha1h s0, s1
    // CHECK:       error: instruction requires: sha2
    // CHECK-NEXT:  sha1h s0, s1
    // CHECK-NEXT:  ^

    .cpu generic+nosha3
    sha512h q0, q1, v2.2d
    // CHECK:       error: instruction requires: sha3
    // CHECK-NEXT:  sha512h q0, q1, v2.2d
    // CHECK-NEXT:  ^

    .cpu generic+nosm4
    sm4e v2.4s, v15.4s
    // CHECK:       error: instruction requires: sm4
    // CHECK-NEXT:  sm4e v2.4s, v15.4s
    // CHECK-NEXT:  ^
