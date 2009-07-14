int top_var;

void top_func_decl(int param1);

void top_func_def(int param2) {
  int local_var1;
  for (int for_var = 100; for_var < 500; ++for_var) {
    int local_var2 = for_var + 1;
  }
}

struct S {
  int field_var;
};
