/* RUN: %clang_cc1 %s -emit-llvm -o - | grep zeroinitializer | count 1

The FE must not generate padding here between array elements.  PR 2533. */

typedef struct {
 const char *name;
 int flags;
 union {
   int x;
 } u;
} OptionDef;

const OptionDef options[] = {
 /* main options */
 { "a", 0, {3} },
 { "b", 0, {4} },
 { 0, },
};
