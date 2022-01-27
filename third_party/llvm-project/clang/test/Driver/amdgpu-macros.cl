// REQUIRES: amdgpu-registered-target
// Check that appropriate macros are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

//
// R600-based processors.
//

// RUN: %clang -E -dM -target r600 -mcpu=r600 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R600 %s -DCPU=r600
// RUN: %clang -E -dM -target r600 -mcpu=rv630 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R600 %s -DCPU=r600
// RUN: %clang -E -dM -target r600 -mcpu=rv635 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R600 %s -DCPU=r600
// RUN: %clang -E -dM -target r600 -mcpu=r630 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R630 %s -DCPU=r630
// RUN: %clang -E -dM -target r600 -mcpu=rs780 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s -DCPU=rs880
// RUN: %clang -E -dM -target r600 -mcpu=rs880 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s -DCPU=rs880
// RUN: %clang -E -dM -target r600 -mcpu=rv610 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s -DCPU=rs880
// RUN: %clang -E -dM -target r600 -mcpu=rv620 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s -DCPU=rs880
// RUN: %clang -E -dM -target r600 -mcpu=rv670 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV670 %s -DCPU=rv670
// RUN: %clang -E -dM -target r600 -mcpu=rv710 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV710 %s -DCPU=rv710
// RUN: %clang -E -dM -target r600 -mcpu=rv730 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV730 %s -DCPU=rv730
// RUN: %clang -E -dM -target r600 -mcpu=rv740 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV770 %s -DCPU=rv770
// RUN: %clang -E -dM -target r600 -mcpu=rv770 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV770 %s -DCPU=rv770
// RUN: %clang -E -dM -target r600 -mcpu=cedar %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CEDAR %s -DCPU=cedar
// RUN: %clang -E -dM -target r600 -mcpu=palm %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CEDAR %s -DCPU=cedar
// RUN: %clang -E -dM -target r600 -mcpu=cypress %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CYPRESS %s -DCPU=cypress
// RUN: %clang -E -dM -target r600 -mcpu=hemlock %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CYPRESS %s -DCPU=cypress
// RUN: %clang -E -dM -target r600 -mcpu=juniper %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,JUNIPER %s -DCPU=juniper
// RUN: %clang -E -dM -target r600 -mcpu=redwood %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,REDWOOD %s -DCPU=redwood
// RUN: %clang -E -dM -target r600 -mcpu=sumo %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,SUMO %s -DCPU=sumo
// RUN: %clang -E -dM -target r600 -mcpu=sumo2 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,SUMO %s -DCPU=sumo
// RUN: %clang -E -dM -target r600 -mcpu=barts %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,BARTS %s -DCPU=barts
// RUN: %clang -E -dM -target r600 -mcpu=caicos %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAICOS %s -DCPU=caicos
// RUN: %clang -E -dM -target r600 -mcpu=aruba %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAYMAN %s -DCPU=cayman
// RUN: %clang -E -dM -target r600 -mcpu=cayman %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAYMAN %s -DCPU=cayman
// RUN: %clang -E -dM -target r600 -mcpu=turks %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,TURKS %s -DCPU=turks

// ARCH-R600-NOT:    #define FP_FAST_FMA 1
// ARCH-R600-NOT:    #define FP_FAST_FMAF 1

// ARCH-R600-DAG: #define __AMDGPU__ 1
// ARCH-R600-DAG: #define __AMD__ 1

// R600-NOT:    #define __HAS_FMAF__ 1
// R630-NOT:    #define __HAS_FMAF__ 1
// RS880-NOT:   #define __HAS_FMAF__ 1
// RV670-NOT:   #define __HAS_FMAF__ 1
// RV710-NOT:   #define __HAS_FMAF__ 1
// RV730-NOT:   #define __HAS_FMAF__ 1
// RV770-NOT:   #define __HAS_FMAF__ 1
// CEDAR-NOT:   #define __HAS_FMAF__ 1
// CYPRESS-DAG: #define __HAS_FMAF__ 1
// JUNIPER-NOT: #define __HAS_FMAF__ 1
// REDWOOD-NOT: #define __HAS_FMAF__ 1
// SUMO-NOT:    #define __HAS_FMAF__ 1
// BARTS-NOT:   #define __HAS_FMAF__ 1
// CAICOS-NOT:  #define __HAS_FMAF__ 1
// CAYMAN-DAG:  #define __HAS_FMAF__ 1
// TURKS-NOT:   #define __HAS_FMAF__ 1

// ARCH-R600-NOT:    #define __HAS_FP64__ 1
// ARCH-R600-NOT:    #define __HAS_LDEXPF__ 1

// ARCH-R600-DAG: #define __R600__ 1

// ARCH-R600-DAG:    #define __[[CPU]]__ 1

//
// AMDGCN-based processors.
//

// RUN: %clang -E -dM -target amdgcn -mcpu=gfx600 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx600
// RUN: %clang -E -dM -target amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx600
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx601 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,SLOW_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx601
// RUN: %clang -E -dM -target amdgcn -mcpu=pitcairn %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx601
// RUN: %clang -E -dM -target amdgcn -mcpu=verde %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx601
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx602 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx602
// RUN: %clang -E -dM -target amdgcn -mcpu=hainan %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx602
// RUN: %clang -E -dM -target amdgcn -mcpu=oland %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx602
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx700 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx700
// RUN: %clang -E -dM -target amdgcn -mcpu=kaveri %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx700
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx701 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx701
// RUN: %clang -E -dM -target amdgcn -mcpu=hawaii %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx701
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx702 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx702
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx703 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx703
// RUN: %clang -E -dM -target amdgcn -mcpu=kabini %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx703
// RUN: %clang -E -dM -target amdgcn -mcpu=mullins %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx703
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx704 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx704
// RUN: %clang -E -dM -target amdgcn -mcpu=bonaire %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx704
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx705 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx705
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx801 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx801
// RUN: %clang -E -dM -target amdgcn -mcpu=carrizo %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx801
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx802
// RUN: %clang -E -dM -target amdgcn -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx802
// RUN: %clang -E -dM -target amdgcn -mcpu=tonga %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx802
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx803
// RUN: %clang -E -dM -target amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx803
// RUN: %clang -E -dM -target amdgcn -mcpu=polaris10 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx803
// RUN: %clang -E -dM -target amdgcn -mcpu=polaris11 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx803
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx805 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx805
// RUN: %clang -E -dM -target amdgcn -mcpu=tongapro %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx805
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx810 %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx810
// RUN: %clang -E -dM -target amdgcn -mcpu=stoney %s 2>&1 | FileCheck --check-prefix=ARCH-GCN %s -DWAVEFRONT_SIZE=64 -DCPU=gfx810
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx900
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx902 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx902
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx904 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx904
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx906
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx908
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx909 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx909
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx90a
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx90c %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=64 -DCPU=gfx90c
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1010
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1011 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1011
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1012 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1012
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1013 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1013
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1030 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1030
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1031 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1031
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1032 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1032
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1033 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1033
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1034 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1034
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1035 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,FAST_FMAF %s -DWAVEFRONT_SIZE=32 -DCPU=gfx1035

// ARCH-GCN-DAG: #define FP_FAST_FMA 1

// FAST_FMAF-DAG: #define FP_FAST_FMAF 1
// SLOW_FMAF-NOT: #define FP_FAST_FMAF 1

// ARCH-GCN-DAG: #define __AMDGCN__ 1
// ARCH-GCN-DAG: #define __AMDGPU__ 1
// ARCH-GCN-DAG: #define __AMD__ 1
// ARCH-GCN-DAG: #define __HAS_FMAF__ 1
// ARCH-GCN-DAG: #define __HAS_FP64__ 1
// ARCH-GCN-DAG: #define __HAS_LDEXPF__ 1
// ARCH-GCN-DAG: #define __[[CPU]]__ 1
// ARCH-GCN-DAG: #define __amdgcn_processor__ "[[CPU]]"
// ARCH-GCN-DAG: #define __AMDGCN_WAVEFRONT_SIZE [[WAVEFRONT_SIZE]]

// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 -mwavefrontsize64 \
// RUN:   %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 -mwavefrontsize64 \
// RUN:   %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 -mwavefrontsize64 \
// RUN:   -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 -mwavefrontsize64 \
// RUN:   -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVE32 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 -mno-wavefrontsize64 \
// RUN:   -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 -mno-wavefrontsize64 \
// RUN:   -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVE64 %s
// WAVE64-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// WAVE32-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32
