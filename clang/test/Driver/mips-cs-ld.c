// Check ld invocations on Mentor Graphics MIPS toolchain.
//
// = Big-endian, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32 %s
// CHECK-BE-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-32: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib/crt1.o"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib/crti.o"
// CHECK-BE-HF-32: "[[TC]]/crtbegin.o"
// CHECK-BE-HF-32: "-L[[TC]]"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/lib"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/usr/lib"
// CHECK-BE-HF-32: "[[TC]]/crtend.o"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib/crtn.o"
//
// = Big-endian, hard float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mips16 \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-16 %s
// CHECK-BE-HF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-16: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/mips16"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib/crt1.o"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib/crti.o"
// CHECK-BE-HF-16: "[[TC]]/mips16/crtbegin.o"
// CHECK-BE-HF-16: "-L[[TC]]/mips16"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16"
// CHECK-BE-HF-16: "-L[[TC]]"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/lib"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib"
// CHECK-BE-HF-16: "[[TC]]/mips16/crtend.o"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib/crtn.o"
//
// = Big-endian, hard float, mmicromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mmicromips \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-MICRO %s
// CHECK-BE-HF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-MICRO: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/micromips"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib/crt1.o"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib/crti.o"
// CHECK-BE-HF-MICRO: "[[TC]]/micromips/crtbegin.o"
// CHECK-BE-HF-MICRO: "-L[[TC]]/micromips"
// CHECK-BE-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips"
// CHECK-BE-HF-MICRO: "-L[[TC]]"
// CHECK-BE-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/lib"
// CHECK-BE-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib"
// CHECK-BE-HF-MICRO: "[[TC]]/micromips/crtend.o"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib/crtn.o"
//
// = Big-endian, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-32 %s
// CHECK-BE-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-32: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/soft-float"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib/crt1.o"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib/crti.o"
// CHECK-BE-SF-32: "[[TC]]/soft-float/crtbegin.o"
// CHECK-BE-SF-32: "-L[[TC]]/soft-float"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/soft-float"
// CHECK-BE-SF-32: "-L[[TC]]"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/lib"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib"
// CHECK-BE-SF-32: "[[TC]]/soft-float/crtend.o"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib/crtn.o"
//
// = Big-endian, soft float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float -mips16 \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-16 %s
// CHECK-BE-SF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-16: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/mips16/soft-float"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib/crt1.o"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib/crti.o"
// CHECK-BE-SF-16: "[[TC]]/mips16/soft-float/crtbegin.o"
// CHECK-BE-SF-16: "-L[[TC]]/mips16/soft-float"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16/soft-float"
// CHECK-BE-SF-16: "-L[[TC]]"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/lib"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib"
// CHECK-BE-SF-16: "[[TC]]/mips16/soft-float/crtend.o"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib/crtn.o"
//
// = Big-endian, soft float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float -mmicromips \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-MICRO %s
// CHECK-BE-SF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-MICRO: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/micromips/soft-float"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib/crt1.o"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib/crti.o"
// CHECK-BE-SF-MICRO: "[[TC]]/micromips/soft-float/crtbegin.o"
// CHECK-BE-SF-MICRO: "-L[[TC]]/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-L[[TC]]"
// CHECK-BE-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/lib"
// CHECK-BE-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib"
// CHECK-BE-SF-MICRO: "[[TC]]/micromips/soft-float/crtend.o"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib/crtn.o"
//
// = Big-endian, hard float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips64-linux-gnu \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64 %s
// CHECK-BE-HF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64/crt1.o"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64/crti.o"
// CHECK-BE-HF-64: "[[TC]]/64/crtbegin.o"
// CHECK-BE-HF-64: "-L[[TC]]/64"
// CHECK-BE-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64"
// CHECK-BE-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/lib/../lib64"
// CHECK-BE-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64"
// CHECK-BE-HF-64: "-L[[TC]]"
// CHECK-BE-HF-64: "[[TC]]/64/crtend.o"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64/crtn.o"
//
// = Big-endian, soft float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips64-linux-gnu -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64 %s
// CHECK-BE-SF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-64: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/soft-float"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64/crt1.o"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64/crti.o"
// CHECK-BE-SF-64: "[[TC]]/soft-float/64/crtbegin.o"
// CHECK-BE-SF-64: "-L[[TC]]/soft-float/64"
// CHECK-BE-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64/soft-float"
// CHECK-BE-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/lib/../lib64"
// CHECK-BE-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64"
// CHECK-BE-SF-64: "-L[[TC]]"
// CHECK-BE-SF-64: "[[TC]]/soft-float/64/crtend.o"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64/crtn.o"
//
// = Little-endian, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mhard-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-32 %s
// CHECK-EL-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-32: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/el"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib/crt1.o"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib/crti.o"
// CHECK-EL-HF-32: "[[TC]]/el/crtbegin.o"
// CHECK-EL-HF-32: "-L[[TC]]/el"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/el"
// CHECK-EL-HF-32: "-L[[TC]]"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/lib"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib"
// CHECK-EL-HF-32: "[[TC]]/el/crtend.o"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib/crtn.o"
//
// = Little-endian, hard float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mips16 \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-16 %s
// CHECK-EL-HF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-16: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/mips16/el"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib/crt1.o"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib/crti.o"
// CHECK-EL-HF-16: "[[TC]]/mips16/el/crtbegin.o"
// CHECK-EL-HF-16: "-L[[TC]]/mips16/el"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16/el"
// CHECK-EL-HF-16: "-L[[TC]]"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/lib"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib"
// CHECK-EL-HF-16: "[[TC]]/mips16/el/crtend.o"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib/crtn.o"
//
// = Little-endian, hard float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mmicromips \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-MICRO %s
// CHECK-EL-HF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-MICRO: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/micromips/el"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib/crt1.o"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib/crti.o"
// CHECK-EL-HF-MICRO: "[[TC]]/micromips/el/crtbegin.o"
// CHECK-EL-HF-MICRO: "-L[[TC]]/micromips/el"
// CHECK-EL-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips/el"
// CHECK-EL-HF-MICRO: "-L[[TC]]"
// CHECK-EL-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/lib"
// CHECK-EL-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib"
// CHECK-EL-HF-MICRO: "[[TC]]/micromips/el/crtend.o"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib/crtn.o"
//
// = Little-endian, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mfloat-abi=soft \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-32 %s
// CHECK-EL-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-32: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/soft-float/el"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib/crt1.o"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib/crti.o"
// CHECK-EL-SF-32: "[[TC]]/soft-float/el/crtbegin.o"
// CHECK-EL-SF-32: "-L[[TC]]/soft-float/el"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/soft-float/el"
// CHECK-EL-SF-32: "-L[[TC]]"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/lib"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib"
// CHECK-EL-SF-32: "[[TC]]/soft-float/el/crtend.o"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib/crtn.o"
//
// = Little-endian, soft float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mips16 -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-16 %s
// CHECK-EL-SF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-16: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib/crt1.o"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib/crti.o"
// CHECK-EL-SF-16: "[[TC]]/mips16/soft-float/el/crtbegin.o"
// CHECK-EL-SF-16: "-L[[TC]]/mips16/soft-float/el"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16/soft-float/el"
// CHECK-EL-SF-16: "-L[[TC]]"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/lib"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib"
// CHECK-EL-SF-16: "[[TC]]/mips16/soft-float/el/crtend.o"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib/crtn.o"
//
// = Little-endian, soft float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mmicromips -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-MICRO %s
// CHECK-EL-SF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-MICRO: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib/crt1.o"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib/crti.o"
// CHECK-EL-SF-MICRO: "[[TC]]/micromips/soft-float/el/crtbegin.o"
// CHECK-EL-SF-MICRO: "-L[[TC]]/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-L[[TC]]"
// CHECK-EL-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/lib"
// CHECK-EL-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib"
// CHECK-EL-SF-MICRO: "[[TC]]/micromips/soft-float/el/crtend.o"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib/crtn.o"
//
// = Little-endian, hard float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips64el-linux-gnu \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64 %s
// CHECK-EL-HF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-64: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/el"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64/crt1.o"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64/crti.o"
// CHECK-EL-HF-64: "[[TC]]/el/64/crtbegin.o"
// CHECK-EL-HF-64: "-L[[TC]]/el/64"
// CHECK-EL-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64/el"
// CHECK-EL-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/lib/../lib64"
// CHECK-EL-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64"
// CHECK-EL-HF-64: "-L[[TC]]"
// CHECK-EL-HF-64: "[[TC]]/el/64/crtend.o"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64/crtn.o"
//
// = Little-endian, soft float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips64el-linux-gnu -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64 %s
// CHECK-EL-SF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-64: "--sysroot=[[TC:[^"]+]]/../../../../mips-linux-gnu/libc/soft-float/el"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64/crt1.o"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64/crti.o"
// CHECK-EL-SF-64: "[[TC]]/soft-float/el/64/crtbegin.o"
// CHECK-EL-SF-64: "-L[[TC]]/soft-float/el/64"
// CHECK-EL-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64/soft-float/el"
// CHECK-EL-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/lib/../lib64"
// CHECK-EL-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64"
// CHECK-EL-SF-64: "-L[[TC]]"
// CHECK-EL-SF-64: "[[TC]]/soft-float/el/64/crtend.o"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64/crtn.o"
