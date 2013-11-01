// RUN: %clang -target x86_64-linux-gnu -fcatch-undefined-behavior %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined-trap -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// RUN: %clang -target x86_64-linux-gnu -fsanitize-undefined-trap-on-error -fsanitize=undefined-trap %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP
// CHECK-UNDEFINED-TRAP: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|shift|unreachable|return|vla-bound|alignment|null|object-size|float-cast-overflow|array-bounds|enum|bool),?){14}"}}
// CHECK-UNDEFINED-TRAP: "-fsanitize-undefined-trap-on-error"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED
// CHECK-UNDEFINED: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|function|shift|unreachable|return|vla-bound|alignment|null|vptr|object-size|float-cast-overflow|array-bounds|enum|bool),?){16}"}}

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-DARWIN
// CHECK-UNDEFINED-DARWIN: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|shift|unreachable|return|vla-bound|alignment|null|vptr|object-size|float-cast-overflow|array-bounds|enum|bool),?){15}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=integer %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-INTEGER
// CHECK-INTEGER: "-fsanitize={{((signed-integer-overflow|unsigned-integer-overflow|integer-divide-by-zero|shift),?){4}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread,undefined -fno-thread-sanitizer -fno-sanitize=float-cast-overflow,vptr,bool,enum %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PARTIAL-UNDEFINED
// CHECK-PARTIAL-UNDEFINED: "-fsanitize={{((signed-integer-overflow|integer-divide-by-zero|float-divide-by-zero|function|shift|unreachable|return|vla-bound|alignment|null|object-size|array-bounds),?){12}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address-full %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-FULL
// CHECK-ASAN-FULL: "-fsanitize={{((address|init-order|use-after-return|use-after-scope),?){4}"}}

// RUN: %clang -target x86_64-linux-gnu -fno-sanitize=init-order,use-after-return -fsanitize=address %s -### 2>&1 |  FileCheck %s --check-prefix=CHECK-ASAN-IMPLIED-INIT-ORDER-UAR
// CHECK-ASAN-IMPLIED-INIT-ORDER-UAR: "-fsanitize={{((address|init-order|use-after-return),?){3}"}}

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fno-sanitize=init-order %s -### 2>&1 |  FileCheck %s --check-prefix=CHECK-ASAN-NO-IMPLIED-INIT-ORDER
// CHECK-ASAN-NO-IMPLIED-INIT-ORDER-NOT: init-order

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fno-sanitize=use-after-return %s -### 2>&1 |  FileCheck %s --check-prefix=CHECK-ASAN-NO-IMPLIED-UAR
// CHECK-ASAN-NO-IMPLIED-UAR-NOT: use-after-return

// RUN: %clang -target x86_64-linux-gnu -fcatch-undefined-behavior -fno-sanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-NO-TRAP-ERROR
// CHECK-UNDEFINED-NO-TRAP-ERROR: '-fcatch-undefined-behavior' not allowed with '-fno-sanitize-undefined-trap-on-error'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fcatch-undefined-behavior %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-UNDEF-ERROR
// CHECK-VPTR-UNDEF-ERROR: '-fsanitize=vptr' not allowed with '-fcatch-undefined-behavior'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP-ON-ERROR-UNDEF
// CHECK-UNDEFINED-TRAP-ON-ERROR-UNDEF: '-fsanitize=undefined' not allowed with '-fsanitize-undefined-trap-on-error'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fsanitize-undefined-trap-on-error %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED-TRAP-ON-ERROR-VPTR
// CHECK-UNDEFINED-TRAP-ON-ERROR-VPTR: '-fsanitize=vptr' not allowed with '-fsanitize-undefined-trap-on-error'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-NO-RTTI
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-VPTR-NO-RTTI
// CHECK-VPTR-NO-RTTI: '-fsanitize=vptr' not allowed with '-fno-rtti'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,thread -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANT
// CHECK-SANA-SANT: '-fsanitize=address' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANM
// CHECK-SANA-SANM: '-fsanitize=address' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANT-SANM
// CHECK-SANT-SANM: '-fsanitize=thread' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANM-SANT
// CHECK-SANM-SANT: '-fsanitize=thread' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak,thread -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-SANT
// CHECK-SANL-SANT: '-fsanitize=leak' not allowed with '-fsanitize=thread'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak,memory -pie -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-SANM
// CHECK-SANL-SANM: '-fsanitize=leak' not allowed with '-fsanitize=memory'

// RUN: %clang -target x86_64-linux-gnu -faddress-sanitizer -fthread-sanitizer -fno-rtti %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-TSAN
// CHECK-ASAN-TSAN: '-faddress-sanitizer' not allowed with '-fthread-sanitizer'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=init-order %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-EXTRA-ASAN
// CHECK-ONLY-EXTRA-ASAN: '-fsanitize=init-order' is ignored in absence of '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -Wno-unused-sanitize-argument -fsanitize=init-order %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-WNO-UNUSED-SANITIZE-ARGUMENT
// CHECK-WNO-UNUSED-SANITIZE-ARGUMENT-NOT: '-fsanitize=init-order' is ignored in absence of '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,init-order -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NOWARN-ONLY-EXTRA-ASAN
// CHECK-NOWARN-ONLY-EXTRA-ASAN-NOT: is ignored in absence of '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu -fsanitize-memory-track-origins -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-TRACK-ORIGINS
// CHECK-ONLY-TRACK-ORIGINS: warning: argument unused during compilation: '-fsanitize-memory-track-origins'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTRA-TRACK-ORIGINS
// CHECK-NO-EXTRA-TRACK-ORIGINS-NOT: "-fsanitize-memory-track-origins"

// RUN: %clang -target x86_64-linux-gnu -fsanitize-address-zero-base-shadow -pie %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ONLY-ASAN-ZERO-BASE-SHADOW
// CHECK-ONLY-ASAN-ZERO-BASE-SHADOW: warning: argument unused during compilation: '-fsanitize-address-zero-base-shadow'

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize=alignment -fsanitize=vptr -fno-sanitize=vptr %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -pie %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-memory-track-origins -pie %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=vptr -fno-sanitize=vptr -fsanitize=undefined,address %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-zero-base-shadow -pie %s -### 2>&1
// OK

// RUN: %clang -target x86_64-linux-gnu -fcatch-undefined-behavior -fthread-sanitizer -fno-thread-sanitizer -faddress-sanitizer -fno-address-sanitizer -fbounds-checking -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEPRECATED
// CHECK-DEPRECATED: argument '-fbounds-checking' is deprecated, use '-fsanitize=local-bounds' instead
// CHECK-DEPRECATED: argument '-fno-address-sanitizer' is deprecated, use '-fno-sanitize=address' instead
// CHECK-DEPRECATED: argument '-faddress-sanitizer' is deprecated, use '-fsanitize=address' instead
// CHECK-DEPRECATED: argument '-fno-thread-sanitizer' is deprecated, use '-fno-sanitize=thread' instead
// CHECK-DEPRECATED: argument '-fthread-sanitizer' is deprecated, use '-fsanitize=thread' instead
// CHECK-DEPRECATED: argument '-fcatch-undefined-behavior' is deprecated, use '-fsanitize=undefined-trap -fsanitize-undefined-trap-on-error' instead

// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-NO-PIE
// CHECK-TSAN-NO-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pie-level" "2"
// CHECK-TSAN-NO-PIE: "-pie"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-NO-PIE
// CHECK-MSAN-NO-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pie-level" "2"
// CHECK-MSAN-NO-PIE: "-pie"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-zero-base-shadow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ZERO-BASE-SHADOW-NO-PIE
// CHECK-ASAN-ZERO-BASE-SHADOW-NO-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pie-level" "2"
// CHECK-ASAN-ZERO-BASE-SHADOW-NO-PIE: "-pie"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-address-zero-base-shadow -fno-sanitize-address-zero-base-shadow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN-ZERO-BASE-SHADOW-CANCEL
// CHECK-ASAN-ZERO-BASE-SHADOW-CANCEL-NOT: "-mrelocation-model" "pic" "-pic-level" "2" "-pie-level" "2"
// CHECK-ASAN-ZERO-BASE-SHADOW-CANCEL-NOT: "-pie"

// RUN: %clang -target arm-linux-androideabi -fsanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ANDROID-ASAN-NO-PIE
// CHECK-ANDROID-ASAN-NO-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pie-level" "2"
// CHECK-ANDROID-ASAN-NO-PIE: "-pie"

// RUN: %clang -target arm-linux-androideabi %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ANDROID-NO-ASAN
// CHECK-ANDROID-NO-ASAN: "-mrelocation-model" "static"

// RUN: %clang -target arm-linux-androideabi -fsanitize=address -fsanitize-address-zero-base-shadow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ANDROID-ASAN-ZERO-BASE
// CHECK-ANDROID-ASAN-ZERO-BASE-NOT: argument unused during compilation

// RUN: %clang -target arm-linux-androideabi -fsanitize=address -fno-sanitize-address-zero-base-shadow %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ANDROID-ASAN-NO-ZERO-BASE
// CHECK-ANDROID-ASAN-NO-ZERO-BASE: '-fno-sanitize-address-zero-base-shadow' not allowed with '-fsanitize=address'

// RUN: %clang -target x86_64-linux-gnu %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER
// RUN: %clang -target x86_64-linux-gnu %s -fno-sanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER
// RUN: %clang -target x86_64-linux-gnu %s -fno-sanitize-recover -fsanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-RECOVER
// RUN: %clang -target x86_64-linux-gnu %s -fsanitize-recover -fno-sanitize-recover -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RECOVER
// CHECK-RECOVER-NOT: sanitize-recover
// CHECK-NO-RECOVER: "-fno-sanitize-recover"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL
// CHECK-SANL: "-fsanitize=leak"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA
// CHECK-SANA-SANL-NO-SANA: "-fsanitize=leak"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN
// CHECK-MSAN: "-fno-assume-sane-operator-new"

// RUN: %clang -target x86_64-linux-gnu -fsanitize=zzz %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG1
// CHECK-DIAG1: unsupported argument 'zzz' to option 'fsanitize='
// CHECK-DIAG1-NOT: unsupported argument 'zzz' to option 'fsanitize='

// RUN: %clang -target i686-linux-gnu -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-X86
// CHECK-MSAN-X86: error: unsupported option '-fsanitize=memory' for target 'i686--linux-gnu'

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-DARWIN
// CHECK-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=memory -fno-sanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-NOMSAN-DARWIN
// CHECK-MSAN-NOMSAN-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=memory -fsanitize=thread,memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN-TSAN-MSAN-DARWIN
// CHECK-MSAN-TSAN-MSAN-DARWIN: unsupported option '-fsanitize=thread,memory' for target 'x86_64-apple-darwin10'
// CHECK-MSAN-TSAN-MSAN-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=thread,memory -fsanitize=memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN-MSAN-MSAN-DARWIN
// CHECK-TSAN-MSAN-MSAN-DARWIN: unsupported option '-fsanitize=memory' for target 'x86_64-apple-darwin10'
// CHECK-TSAN-MSAN-MSAN-DARWIN: unsupported option '-fsanitize=thread' for target 'x86_64-apple-darwin10'
// CHECK-TSAN-MSAN-MSAN-DARWIN-NOT: unsupported option

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=function %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSAN-DARWIN
// CHECK-FSAN-DARWIN: unsupported option '-fsanitize=function' for target 'x86_64-apple-darwin10'

// RUN: %clang -target x86_64-apple-darwin10 -fsanitize=function -fsanitize=undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FSAN-UBSAN-DARWIN
// CHECK-FSAN-UBSAN-DARWIN: unsupported option '-fsanitize=function' for target 'x86_64-apple-darwin10'
