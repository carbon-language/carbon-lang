
typedef union {
   long (*ap)[4];
} ptrs;

void DoAssignIteration() {
  ptrs abase;
  abase.ap+=27;
  Assignment(*abase.ap);
}


