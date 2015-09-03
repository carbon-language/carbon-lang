// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-DEFAULT %s
// TRAP-DEFAULT: -cc1as
// TRAP-DEFAULT-NOT: "-target-feature" "-use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--trap 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-ON %s
// TRAP-ON: -cc1as
// TRAP-ON: "-target-feature" "+use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--break 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-OFF %s
// TRAP-OFF: -cc1as
// TRAP-OFF: "-target-feature" "-use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--trap,--break 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-BOTH-TRAP-FIRST %s
// TRAP-BOTH-TRAP-FIRST: -cc1as
// TRAP-BOTH-TRAP-FIRST: "-target-feature" "+use-tcc-in-div" "-target-feature" "-use-tcc-in-div"

// RUN: %clang -target mips-linux-gnu -### -fintegrated-as -c %s -Wa,--break,--trap 2>&1 | \
// RUN:   FileCheck -check-prefix=TRAP-BOTH-BREAK-FIRST %s
// TRAP-BOTH-BREAK-FIRST: -cc1as
// TRAP-BOTH-BREAK-FIRST: "-target-feature" "-use-tcc-in-div" "-target-feature" "+use-tcc-in-div"
