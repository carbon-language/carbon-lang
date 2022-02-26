; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=SM20
; RUN: llc < %s -march=nvptx -mcpu=sm_21 | FileCheck %s --check-prefix=SM21
; RUN: llc < %s -march=nvptx -mcpu=sm_30 | FileCheck %s --check-prefix=SM30
; RUN: llc < %s -march=nvptx -mcpu=sm_32 | FileCheck %s --check-prefix=SM32
; RUN: llc < %s -march=nvptx -mcpu=sm_35 | FileCheck %s --check-prefix=SM35
; RUN: llc < %s -march=nvptx -mcpu=sm_37 | FileCheck %s --check-prefix=SM37
; RUN: llc < %s -march=nvptx -mcpu=sm_50 | FileCheck %s --check-prefix=SM50
; RUN: llc < %s -march=nvptx -mcpu=sm_52 | FileCheck %s --check-prefix=SM52
; RUN: llc < %s -march=nvptx -mcpu=sm_53 | FileCheck %s --check-prefix=SM53
; RUN: llc < %s -march=nvptx -mcpu=sm_60 | FileCheck %s --check-prefix=SM60
; RUN: llc < %s -march=nvptx -mcpu=sm_61 | FileCheck %s --check-prefix=SM61
; RUN: llc < %s -march=nvptx -mcpu=sm_62 | FileCheck %s --check-prefix=SM62
; RUN: llc < %s -march=nvptx -mcpu=sm_70 | FileCheck %s --check-prefix=SM70
; RUN: llc < %s -march=nvptx -mcpu=sm_75 | FileCheck %s --check-prefix=SM75
; RUN: llc < %s -march=nvptx -mcpu=sm_80 | FileCheck %s --check-prefix=SM80
; RUN: llc < %s -march=nvptx -mcpu=sm_86 | FileCheck %s --check-prefix=SM86

; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=SM20
; RUN: llc < %s -march=nvptx64 -mcpu=sm_21 | FileCheck %s --check-prefix=SM21
; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s --check-prefix=SM30
; RUN: llc < %s -march=nvptx64 -mcpu=sm_32 | FileCheck %s --check-prefix=SM32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix=SM35
; RUN: llc < %s -march=nvptx64 -mcpu=sm_37 | FileCheck %s --check-prefix=SM37
; RUN: llc < %s -march=nvptx64 -mcpu=sm_50 | FileCheck %s --check-prefix=SM50
; RUN: llc < %s -march=nvptx64 -mcpu=sm_52 | FileCheck %s --check-prefix=SM52
; RUN: llc < %s -march=nvptx64 -mcpu=sm_53 | FileCheck %s --check-prefix=SM53
; RUN: llc < %s -march=nvptx64 -mcpu=sm_60 | FileCheck %s --check-prefix=SM60
; RUN: llc < %s -march=nvptx64 -mcpu=sm_61 | FileCheck %s --check-prefix=SM61
; RUN: llc < %s -march=nvptx64 -mcpu=sm_62 | FileCheck %s --check-prefix=SM62
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 | FileCheck %s --check-prefix=SM70
; RUN: llc < %s -march=nvptx64 -mcpu=sm_75 | FileCheck %s --check-prefix=SM75
; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 | FileCheck %s --check-prefix=SM80
; RUN: llc < %s -march=nvptx64 -mcpu=sm_86 | FileCheck %s --check-prefix=SM86

; SM30: .version 3.2
; SM32: .version 4.0
; SM35: .version 3.2
; SM37: .version 4.1
; SM50: .version 4.0
; SM52: .version 4.1
; SM53: .version 4.2
; SM60: .version 5.0
; SM61: .version 5.0
; SM62: .version 5.0
; SM70: .version 6.0
; SM75: .version 6.3
; SM80: .version 7.0
; SM86: .version 7.1

; SM20: .target sm_20
; SM21: .target sm_21
; SM30: .target sm_30
; SM32: .target sm_32
; SM35: .target sm_35
; SM37: .target sm_37
; SM50: .target sm_50
; SM52: .target sm_52
; SM53: .target sm_53
; SM60: .target sm_60
; SM61: .target sm_61
; SM62: .target sm_62
; SM70: .target sm_70
; SM75: .target sm_75
; SM80: .target sm_80
; SM86: .target sm_86
