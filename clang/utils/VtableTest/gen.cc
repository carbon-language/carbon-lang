#include <stdio.h>
#include <stdlib.h>

#define N_FIELDS 7
#define N_FUNCS 128
#define FUNCSPACING 10
#define N_STRUCTS 300 /* 1280 */
// FIXME: Need final overrider logic for 3 or more if we turn on virtual base
//        class dups
#define N_BASES 30 /* 30 */

const char *simple_types[] = { "bool", "char", "short", "int", "float",
			       "double", "long double", "wchar_t", "void *",
			       "char *"
};

void gl(const char *c) {
  printf("%s\n", c);
}

void g(const char *c) {
  printf("%s", c);
}

void g(int i) {
  printf("%d", i);
}

int vfn = 0;
char base_present[N_STRUCTS][N_STRUCTS];

bool is_ambiguous(int s, int base) {
  for (int i = 0; i < N_STRUCTS; ++i) {
    if ((base_present[base][i] & base_present[s][i])
	// FIXME: todo this, we need final overrider additions
	/*== 1 */)
      return true;
  }
  return false;
}

void add_bases(int s, int base) {
  for (int i = 0; i < N_STRUCTS; ++i)
    base_present[s][i] |= base_present[base][i];
}

void gs(int s) {
  bool polymorphic = false;

  static int bases[N_BASES];
  int i_bases = random() % (N_BASES*2);
  if (i_bases >= N_BASES)
    // PARAM: 1/2 of all clases should have no bases
    i_bases = 0;
  int n_bases = 0;
  bool first_base = true;
  
  // PARAM: 3/4 of all should be class, the rest are structs
  if (random() % 4 == 0)
    g("struct s");
  else
    g("class s");
  g(s);
  int old_base = -1;
  if (s == 0)
    i_bases = 0;
  while (i_bases) {
    --i_bases;
    int base = random() % s;
    if (!base_present[s][base]) {
      if (is_ambiguous(s, base))
	continue;
      if (first_base) {
	first_base = false;
	g(": ");
      } else
	g(", ");
      int base_type = 1;
      if (random()%8 == 0) {
	// PARAM: 1/8th the bases are virtual
	g("virtual ");
	polymorphic = true;
	base_type = 3;
      }
      switch (random()%8) {
      case 0:
      case 1:
      case 2:
      case 3:
	break;
      case 4:
      case 5:
	g("public "); break;
      case 6:
	g("private "); break;
      case 7:
	g("protected "); break;
      }
      g("s");
      add_bases(s, base);
      bases[n_bases] = base;
      base_present[s][base] = base_type;
      ++n_bases;
      g(base);
      old_base = base;
    }
  }
  gl(" {");

  /* Fields */
  int n_fields = random() % (N_FIELDS*4);
  // PARAM: 3/4 of all structs should have no members
  if (n_fields >= N_FIELDS)
    n_fields = 0;
  for (int i = 0; i < n_fields; ++i) {
    int t = random() % (sizeof(simple_types) / sizeof(simple_types[0]));
    g("  "); g(simple_types[t]); g(" field"); g(i); gl(";");
  }

  /* Virtual functions */
  static int funcs[N_FUNCS];
  int n_funcs = random() % N_FUNCS;
  int old_func = -1;
  for (int i = 0; i < n_funcs; ++i) {
    int fn = old_func + random() % FUNCSPACING + 1;
    funcs[i] = fn;
    g("  virtual void fun"); g(fn); g("(char *t) { mix((char *)this - t, "); g(++vfn); gl("); }");
    old_func = fn;
  }

  gl("public:");
  gl("  void calc(char *t) {");

  // mix in the type number
  g("    mix("); g(s); gl(");");
  // mix in the size
  g("    mix(sizeof (s"); g(s); gl("));");
  // mix in the this offset
  gl("    mix((char *)this - t);");
  if (n_funcs)
    polymorphic = true;
  if (polymorphic) {
    // mix in offset to the complete object under construction
    gl("    mix(t - (char *)dynamic_cast<void*>(this));");
  }

  /* check base layout and overrides */
  for (int i = 0; i < n_bases; ++i) {
    g("    calc_s"); g(bases[i]); gl("(t);");
  }

  if (polymorphic) {
    /* check dynamic_cast to each direct base */
    for (int i = 0; i < n_bases; ++i) {
      g("    if ((char *)dynamic_cast<s"); g(bases[i]); gl("*>(this))");
      g("      mix(t - (char *)dynamic_cast<s"); g(bases[i]); gl("*>(this));");
      gl("    else mix(666);");
    }
  }

  /* check field layout */
  for (int i = 0; i < n_fields; ++i) {
    g("    mix((char *)&field"); g(i); gl(" - (char *)this);");
  }
  if (n_fields == 0)
    gl("    mix(42);");

  /* check functions */
  for (int i = 0; i < n_funcs; ++i) {
    g("    fun"); g(funcs[i]); gl("(t);");
  }
  if (n_funcs == 0)
    gl("    mix(13);");

  gl("  }");

  // default ctor
  g("  s"); g(s); g("() ");
  first_base = true;
  for (int i = 0; i < n_bases; ++i) {
    if (first_base) {
      g(": ");
      first_base = false;
    } else
      g(", ");
    g("s"); g(bases[i]); g("((char *)this)");
  }
  gl(" { calc((char *)this); }");
  g("  ~s"); g(s); gl("() { calc((char *)this); }");

 // ctor with this to the complete object
  g("  s"); g(s); gl("(char *t) { calc(t); }");
  g("  void calc_s"); g(s); gl("(char *t) { calc(t); }");
  g("} a"); g(s); gl(";");
}

main(int argc, char **argv) {
  unsigned seed = 0;
  char state[16];
  if (argc > 1)
    seed = atol(argv[1]);

  initstate(seed, state, sizeof(state));
  gl("extern \"C\" int printf(const char *...);");
  gl("");
  gl("long long sum;");
  gl("void mix(long long i) {");
  // If this ever becomes too slow, we can remove this after we improve the
  // mixing function
  gl("  printf(\"%llx\\n\", i);");
  gl("  sum += ((sum ^ i) << 3) + (sum<<1) - i;");
  gl("}");
  gl("void mix(long long i1, long long i2) { mix(i1); mix(i2); }");
  gl("");
  // PARAM: Randomly size testcases or large testcases?
  int n_structs = /* random() % */ N_STRUCTS;
  for (int i = 0; i < n_structs; ++i)
    gs(i);
  gl("int main() {");
  gl("  printf(\"%llx\\n\", sum);");
  gl("}");
  return 0;
}
