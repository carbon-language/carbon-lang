// RUN: %clang %s -O0 -emit-llvm -S -o - | grep 'call.*rb_define_global_function'
// This should call rb_define_global_function, not rb_f_chop.

void rb_define_global_function (const char*,void(*)(),int);
static void rb_f_chop();
void Init_String() {
  rb_define_global_function("chop", rb_f_chop, 0);
}
static void rb_f_chop() {
}

