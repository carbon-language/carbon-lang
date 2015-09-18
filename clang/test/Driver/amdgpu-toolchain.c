// RUN: %clang -### -target amdgcn--amdhsa -x assembler -mcpu=kaveri %s
// RUN: %clang -### -target amdgcn--amdhsa -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// AS_LINK-LABEL: clang
// AS_LINK: "-cc1as"
// AS_LINK-LABEL: lld
// AS_LINK: "-flavor" "gnu" "-target" "amdgcn--amdhsa"
