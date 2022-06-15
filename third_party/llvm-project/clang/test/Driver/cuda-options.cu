// Tests CUDA compilation pipeline construction in Driver.
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Simple compilation case. Compile device-side to PTX assembly and make sure
// we use it on the host side.
// RUN: %clang -### -target x86_64-linux-gnu -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix NOLINK %s

// Typical compilation + link case.
// RUN: %clang -### -target x86_64-linux-gnu %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// Verify that --cuda-host-only disables device-side compilation, but doesn't
// disable host-side compilation/linking.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// Verify that --cuda-device-only disables host-side compilation and linking.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// Check that the last of --cuda-compile-host-device, --cuda-host-only, and
// --cuda-device-only wins.

// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:    --cuda-host-only %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// RUN: %clang -### -target x86_64-linux-gnu --cuda-compile-host-device \
// RUN:    --cuda-host-only %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only \
// RUN:    --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// RUN: %clang -### -target x86_64-linux-gnu --cuda-compile-host-device \
// RUN:    --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only \
// RUN:   --cuda-compile-host-device %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-compile-host-device %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// Verify that --cuda-gpu-arch option passes the correct GPU architecture to
// device compilation.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_30 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix DEVICE-SM30 -check-prefix HOST \
// RUN:    -check-prefix INCLUDES-DEVICE -check-prefix NOLINK %s

// Verify that there is one device-side compilation per --cuda-gpu-arch args
// and that all results are included on the host side.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes DEVICE,DEVICE-NOSAVE,DEVICE2 \
// RUN:             -check-prefixes DEVICE-SM30,DEVICE2-SM35 \
// RUN:             -check-prefixes INCLUDES-DEVICE,INCLUDES-DEVICE2 \
// RUN:             -check-prefixes HOST,HOST-NOSAVE,NOLINK %s

// Verify that device-side results are passed to the correct tool when
// -save-temps is used.
// RUN: %clang -### -target x86_64-linux-gnu -save-temps -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-SAVE \
// RUN:    -check-prefix HOST -check-prefix HOST-SAVE -check-prefix NOLINK %s

// Verify that device-side results are passed to the correct tool when
// -fno-integrated-as is used.
// RUN: %clang -### -target x86_64-linux-gnu -fno-integrated-as -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix HOST-NOSAVE \
// RUN:    -check-prefix HOST-AS -check-prefix NOLINK %s

// Verify that --[no-]cuda-gpu-arch arguments are handled correctly.
// a) --no-cuda-gpu-arch=X negates preceding --cuda-gpu-arch=X
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-gpu-arch=sm_50 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-gpu-arch=sm_50 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM30,NOARCH-SM35,NOARCH-SM50 %s

// b) --no-cuda-gpu-arch=X negates more than one preceding --cuda-gpu-arch=X
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-gpu-arch=sm_50 --cuda-gpu-arch=sm_50 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-gpu-arch=sm_50 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM30,NOARCH-SM35,NOARCH-SM50 %s

// c) if --no-cuda-gpu-arch=X negates all preceding --cuda-gpu-arch=X
//    we default to sm_35 -- same as if no --cuda-gpu-arch were passed.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-gpu-arch=sm_50 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-gpu-arch=sm_50 --no-cuda-gpu-arch=sm_30 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes NOARCH-SM30,ARCH-SM35,NOARCH-SM50 %s

// d) --no-cuda-gpu-arch=X is a no-op if there's no preceding --cuda-gpu-arch=X
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30\
// RUN:   --no-cuda-gpu-arch=sm_50 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM30,ARCH-SM35,NOARCH-SM50 %s

// e) --no-cuda-gpu-arch=X does not affect following --cuda-gpu-arch=X
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --no-cuda-gpu-arch=sm_50 --no-cuda-gpu-arch=sm_30 \
// RUN:   --cuda-gpu-arch=sm_50 --cuda-gpu-arch=sm_30 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM30,NOARCH-SM35,ARCH-SM50 %s

// f) --no-cuda-gpu-arch=all negates all preceding --cuda-gpu-arch=X
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-gpu-arch=all \
// RUN:   --cuda-gpu-arch=sm_50 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes NOARCH-SM30,NOARCH-SM35,ARCH-SM50 %s

// g) There's no --cuda-gpu-arch=all
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   --cuda-gpu-arch=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCHALLERROR %s


// Verify that --[no-]cuda-include-ptx arguments are handled correctly.
// a) by default we're including PTX for all GPUs.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM35,PTX-SM30 %s

// b) --no-cuda-include-ptx=all disables PTX inclusion for all GPUs
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-include-ptx=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,NOPTX-SM35,NOPTX-SM30 %s

// c) --no-cuda-include-ptx=sm_XX disables PTX inclusion for that GPU only.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-include-ptx=sm_35 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,NOPTX-SM35,PTX-SM30 %s
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-include-ptx=sm_30 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM35,NOPTX-SM30 %s

// d) --cuda-include-ptx=all overrides preceding --no-cuda-include-ptx=all
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-include-ptx=all --cuda-include-ptx=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM35,PTX-SM30 %s

// e) --cuda-include-ptx=all overrides preceding --no-cuda-include-ptx=sm_XX
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 \
// RUN:   --no-cuda-include-ptx=sm_30 --cuda-include-ptx=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM35,PTX-SM30 %s

// Verify -flto=thin -fwhole-program-vtables handling. This should result in
// both options being passed to the host compilation, with neither passed to
// the device compilation.
// RUN: %clang -### -target x86_64-linux-gnu -c -flto=thin -fwhole-program-vtables %s 2>&1 \
// RUN: | FileCheck -check-prefixes DEVICE,DEVICE-NOSAVE,HOST,INCLUDES-DEVICE,NOLINK,THINLTOWPD %s
// THINLTOWPD-NOT: error: invalid argument '-fwhole-program-vtables' only allowed with '-flto'

// ARCH-SM30: "-cc1"{{.*}}"-target-cpu" "sm_30"
// NOARCH-SM30-NOT: "-cc1"{{.*}}"-target-cpu" "sm_30"
// ARCH-SM35: "-cc1"{{.*}}"-target-cpu" "sm_35"
// NOARCH-SM35-NOT: "-cc1"{{.*}}"-target-cpu" "sm_35"
// ARCH-SM50: "-cc1"{{.*}}"-target-cpu" "sm_50"
// NOARCH-SM50-NOT: "-cc1"{{.*}}"-target-cpu" "sm_50"
// ARCHALLERROR: error: unsupported CUDA gpu architecture: all

// Match device-side preprocessor and compiler phases with -save-temps.
// DEVICE-SAVE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-SAVE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// DEVICE-SAVE-SAME: "-fcuda-is-device"
// DEVICE-SAVE-SAME: "-x" "cuda"

// DEVICE-SAVE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-SAVE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// DEVICE-SAVE-SAME: "-fcuda-is-device"
// DEVICE-SAVE-SAME: "-x" "cuda-cpp-output"

// Match the job that produces PTX assembly.
// DEVICE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-NOSAVE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// THINLTOWPD-NOT: "-flto=thin"
// DEVICE-SAME: "-fcuda-is-device"
// DEVICE-SM30-SAME: "-target-cpu" "sm_30"
// THINLTOWPD-NOT: "-fwhole-program-vtables"
// DEVICE-SAME: "-o" "[[PTXFILE:[^"]*]]"
// DEVICE-NOSAVE-SAME: "-x" "cuda"
// DEVICE-SAVE-SAME: "-x" "ir"

// Match the call to ptxas (which assembles PTX to SASS).
// DEVICE:ptxas
// DEVICE-SM30-DAG: "--gpu-name" "sm_30"
// DEVICE-DAG: "--output-file" "[[CUBINFILE:[^"]*]]"
// DEVICE-DAG: "[[PTXFILE]]"

// Match another device-side compilation.
// DEVICE2: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE2-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// DEVICE2-SAME: "-fcuda-is-device"
// DEVICE2-SM35-SAME: "-target-cpu" "sm_35"
// DEVICE2-SAME: "-o" "[[PTXFILE2:[^"]*]]"
// DEVICE2-SAME: "-x" "cuda"

// Match another call to ptxas.
// DEVICE2: ptxas
// DEVICE2-SM35-DAG: "--gpu-name" "sm_35"
// DEVICE2-DAG: "--output-file" "[[CUBINFILE2:[^"]*]]"
// DEVICE2-DAG: "[[PTXFILE2]]"

// Match no device-side compilation.
// NODEVICE-NOT: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// NODEVICE-NOT: "-fcuda-is-device"

// INCLUDES-DEVICE:fatbinary
// INCLUDES-DEVICE-DAG: "--create" "[[FATBINARY:[^"]*]]"
// INCLUDES-DEVICE-DAG: "--image=profile=sm_{{[0-9]+}},file=[[CUBINFILE]]"
// INCLUDES-DEVICE-DAG: "--image=profile=compute_{{[0-9]+}},file=[[PTXFILE]]"
// INCLUDES-DEVICE2-DAG: "--image=profile=sm_{{[0-9]+}},file=[[CUBINFILE2]]"
// INCLUDES-DEVICE2-DAG: "--image=profile=compute_{{[0-9]+}},file=[[PTXFILE2]]"

// Match host-side preprocessor job with -save-temps.
// HOST-SAVE: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-SAVE-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// HOST-SAVE-NOT: "-fcuda-is-device"
// HOST-SAVE-SAME: "-x" "cuda"

// Match host-side compilation.
// HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// THINLTOWPD-SAME: "-flto=thin"
// HOST-NOT: "-fcuda-is-device"
// There is only one GPU binary after combining it with fatbinary!
// INCLUDES-DEVICE2-NOT: "-fcuda-include-gpubinary"
// INCLUDES-DEVICE-SAME: "-fcuda-include-gpubinary" "[[FATBINARY]]"
// There is only one GPU binary after combining it with fatbinary.
// INCLUDES-DEVICE2-NOT: "-fcuda-include-gpubinary"
// THINLTOWPD-SAME: "-fwhole-program-vtables"
// HOST-SAME: "-o" "[[HOSTOUTPUT:[^"]*]]"
// HOST-NOSAVE-SAME: "-x" "cuda"
// HOST-SAVE-SAME: "-x" "cuda-cpp-output"

// Match external assembler that uses compilation output.
// HOST-AS: "-o" "{{.*}}.o" "[[HOSTOUTPUT]]"

// Match no GPU code inclusion.
// NOINCLUDES-DEVICE-NOT: "-fcuda-include-gpubinary"

// Match no host compilation.
// NOHOST-NOT: "-cc1" "-triple"
// NOHOST-NOT: "-x" "cuda"

// Match linker.
// LINK: "{{.*}}{{ld|link}}{{(.exe)?}}"
// LINK-SAME: "[[HOSTOUTPUT]]"

// Match no linker.
// NOLINK-NOT: "{{.*}}{{ld|link}}{{(.exe)?}}"

// FATBIN-COMMON:fatbinary
// FATBIN-COMMON: "--create" "[[FATBINARY:[^"]*]]"
// FATBIN-COMMON: "--image=profile=sm_30,file=
// PTX-SM30: "--image=profile=compute_30,file=
// NOPTX-SM30-NOT: "--image=profile=compute_30,file=
// FATBIN-COMMON: "--image=profile=sm_35,file=
// PTX-SM35: "--image=profile=compute_35,file=
// NOPTX-SM35-NOT: "--image=profile=compute_35,file=
