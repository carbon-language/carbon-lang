// RUN: %llvmgcc_only %s -c -o /dev/null |& \
// RUN: egrep {(14|15|22): warning:} |	\
// RUN: wc -l | grep --quiet 3
// XTARGET: darwin,linux
// XFAIL: *
// END.
// Insist upon warnings for inappropriate weak attributes.
// Note the line numbers (14|15|22) embedded in the check.

// O.K.
extern int ext_weak_import __attribute__ ((__weak_import__));

// These are inappropriate, and should generate warnings:
int decl_weak_import __attribute__ ((__weak_import__));
int decl_initialized_weak_import __attribute__ ((__weak_import__)) = 13;

// O.K.
extern int ext_f(void) __attribute__ ((__weak_import__));

// These are inappropriate, and should generate warnings:
int def_f(void) __attribute__ ((__weak_import__));
int __attribute__ ((__weak_import__)) decl_f(void) {return 0;};
