// Some convenient things to return:
static char *g_first_pointer = "I am the first";
static char *g_second_pointer = "I am the second";

// First we have some simple functions that return standard types, ints, floats and doubles.
// We have a function calling a function in a few cases to test that if you stop in the
// inner function then do "up/fin" you get the return value from the outer-most frame.

int 
inner_sint (int value)
{
  return value;
}

int
outer_sint (int value)
{
  int outer_value = 2 * inner_sint (value);
  return outer_value;
}

float
inner_float (float value)
{
  return value;
}

float 
outer_float (float value)
{
  float outer_value = 2 * inner_float(value);
  return outer_value;
}

double 
return_double (double value)
{
  return value;
}

long double 
return_long_double (long double value)
{
  return value;
}

char *
return_pointer (char *value)
{
  return value;
}

struct one_int
{
  int one_field;
};

struct one_int
return_one_int (struct one_int value)
{
  return value;
}

struct two_int
{
  int first_field;
  int second_field;
};

struct two_int
return_two_int (struct two_int value)
{
  return value;
}

struct three_int
{
  int first_field;
  int second_field;
  int third_field;
};

struct three_int
return_three_int (struct three_int value)
{
  return value;
}

struct four_int
{
  int first_field;
  int second_field;
  int third_field;
  int fourth_field;
};

struct four_int
return_four_int (struct four_int value)
{
  return value;
}

struct five_int
{
  int first_field;
  int second_field;
  int third_field;
  int fourth_field;
  int fifth_field;
};

struct five_int
return_five_int (struct five_int value)
{
  return value;
}

struct one_int_one_double
{
  int first_field;
  double second_field;
};

struct one_int_one_double
return_one_int_one_double (struct one_int_one_double value)
{
  return value;
}

struct one_int_one_double_one_int
{
  int one_field;
  double second_field;
  int third_field;
};

struct one_int_one_double_one_int
return_one_int_one_double_one_int (struct one_int_one_double_one_int value)
{
  return value;
}

struct one_short_one_double_one_short
{
  int one_field;
  double second_field;
  int third_field;
};

struct one_short_one_double_one_short
return_one_short_one_double_one_short (struct one_short_one_double_one_short value)
{
  return value;
}

struct three_short_one_float
{
  short one_field;
  short second_field;
  short third_field;
  float fourth_field;
};

struct three_short_one_float
return_three_short_one_float (struct three_short_one_float value)
{
  return value;
}

struct one_int_one_float_one_int
{
  int one_field;
  float second_field;
  int third_field;
};

struct one_int_one_float_one_int
return_one_int_one_float_one_int (struct one_int_one_float_one_int value)
{
  return value;
}

struct one_float_one_int_one_float
{
  float one_field;
  int second_field;
  float third_field;
};

struct one_float_one_int_one_float
return_one_float_one_int_one_float (struct one_float_one_int_one_float value)
{
  return value;
}

struct one_double_two_float
{
  double one_field;
  float second_field;
  float third_field;
};

struct one_double_two_float
return_one_double_two_float (struct one_double_two_float value)
{
  return value;
}

struct two_double
{
  double first_field;
  double second_field;
};

struct two_double
return_two_double (struct two_double value)
{
  return value;
}

struct two_float
{
  float first_field;
  float second_field;
};

struct two_float
return_two_float (struct two_float value)
{
  return value;
}

struct one_int_one_double_packed
{
  int first_field;
  double second_field;
} __attribute__((__packed__));

struct one_int_one_double_packed
return_one_int_one_double_packed (struct one_int_one_double_packed value)
{
  return value;
}

struct one_int_one_long
{
  int first_field;
  long second_field;
};

struct one_int_one_long
return_one_int_one_long (struct one_int_one_long value)
{
  return value;
}

struct one_pointer
{
  char *first_field;
};

struct one_pointer
return_one_pointer (struct one_pointer value)
{
  return value;
}

struct two_pointer
{
  char *first_field;
  char *second_field;
};

struct two_pointer
return_two_pointer (struct two_pointer value)
{
  return value;
}

struct one_float_one_pointer
{
  float first_field;
  char *second_field;
};

struct one_float_one_pointer
return_one_float_one_pointer (struct one_float_one_pointer value)
{
  return value;
}

struct one_int_one_pointer
{
  int first_field;
  char *second_field;
};

struct one_int_one_pointer
return_one_int_one_pointer (struct one_int_one_pointer value)
{
  return value;
}

struct base_one_char {
  char c;
};

struct nested_one_float_three_base {
  float f;
  struct base_one_char b1;
  struct base_one_char b2;
  struct base_one_char b3;
}; // returned in RAX for both SysV and Windows

struct nested_one_float_three_base
return_nested_one_float_three_base (struct nested_one_float_three_base value)
{
  return value;
}

struct double_nested_one_float_one_nested {
  float f;
  struct nested_one_float_three_base ns;
}; // SysV-ABI: returned in XMM0 + RAX
// Windows-ABI: returned in memory

struct double_nested_one_float_one_nested
return_double_nested_one_float_one_nested(struct double_nested_one_float_one_nested value)
{
  return value;
}

struct base_float_struct {
  float f1;
  float f2;
};

struct nested_float_struct {
  double d;
  struct base_float_struct bfs;
}; // SysV-ABI: return in xmm0 + xmm1
// Windows-ABI: returned in memory

struct nested_float_struct
return_nested_float_struct (struct nested_float_struct value)
{
  return value;
}

struct six_double_three_int {
  double d1;  // 8
  double d2;  // 8
  int i1;   // 4
  double d3;  // 8
  double d4;  // 8
  int i2;   // 4
  double d5;  // 8
  double d6;  // 8
  int i3;   // 4
}; // returned in memeory on both SysV and Windows

struct six_double_three_int
return_six_double_three_int (struct six_double_three_int value) {
  return value;
}

typedef float vector_size_float32_8 __attribute__((__vector_size__(8)));
typedef float vector_size_float32_16 __attribute__((__vector_size__(16)));
typedef float vector_size_float32_32 __attribute__((__vector_size__(32)));

typedef float ext_vector_size_float32_2 __attribute__((ext_vector_type(2)));
typedef float ext_vector_size_float32_4 __attribute__((ext_vector_type(4)));
typedef float ext_vector_size_float32_8 __attribute__((ext_vector_type(8)));

vector_size_float32_8
return_vector_size_float32_8 (vector_size_float32_8 value)
{
    return value;
}

vector_size_float32_16
return_vector_size_float32_16 (vector_size_float32_16 value)
{
    return value;
}

vector_size_float32_32
return_vector_size_float32_32 (vector_size_float32_32 value)
{
    return value;
}

ext_vector_size_float32_2
return_ext_vector_size_float32_2 (ext_vector_size_float32_2 value)
{
    return value;
}

ext_vector_size_float32_4
return_ext_vector_size_float32_4 (ext_vector_size_float32_4 value)
{
    return value;
}

ext_vector_size_float32_8
return_ext_vector_size_float32_8 (ext_vector_size_float32_8 value)
{
    return value;
}

class base_class_one_char {
public:
  char c = '!';
}; // returned in RAX for both ABI

base_class_one_char
return_base_class_one_char(base_class_one_char value) {
  return value;
}

class nested_class_float_and_base {
public:
  float f = 0.1;
  base_class_one_char b;
}; // returned in RAX for both ABI

nested_class_float_and_base
return_nested_class_float_and_base(nested_class_float_and_base value) {
  return value;
}

class double_nested_class_float_and_nested {
public:
  float f = 0.2;
  nested_class_float_and_base n;
}; // SysV-ABI: returned in XMM0 + RAX
// Windows-ABI: returned in memory

double_nested_class_float_and_nested
return_double_nested_class_float_and_nested(
    double_nested_class_float_and_nested value) {
  return value;
}

class base_class {
public:
  base_class() {
    c = 'a';
    c2 = 'b';
  }
private:
  char c;
protected:
  char c2;
}; // returned in RAX for both ABI

base_class
return_base_class(base_class value) {
  return value;
}

class sub_class : base_class {
public:
  sub_class() {
    c2 = '&';
    i = 10;
  }
private:
  int i;
}; // size 8; should be returned in RAX
// Since it's in register, lldb won't be able to get the
// fields in base class, expected to fail.

sub_class
return_sub_class(sub_class value) {
  return value;
}

class abstract_class {
public:
  virtual char getChar() = 0;
private:
  int i = 8;
protected:
  char c = '!';
};

class derived_class : abstract_class {
public:
  derived_class() {
    c = '?';
  }
  char getChar() {
    return this->c;
  }
private:
  char c2 = '$';
}; // size: 16; contains non POD member, returned in memory

derived_class
return_derived_class(derived_class value) {
  return value;
}

int 
main ()
{
  int first_int = 123456;
  int second_int = 234567;

  outer_sint (first_int);
  outer_sint (second_int);

  float first_float_value = 12.34;
  float second_float_value = 23.45;

  outer_float (first_float_value);
  outer_float (second_float_value);

  double double_value = -23.45;

  return_double (double_value);

  return_pointer(g_first_pointer);

  long double long_double_value = -3456789.987654321;

  return_long_double (long_double_value);

  // Okay, now the structures:
  return_one_int ((struct one_int) {10});
  return_two_int ((struct two_int) {10, 20});
  return_three_int ((struct three_int) {10, 20, 30});
  return_four_int ((struct four_int) {10, 20, 30, 40});
  return_five_int ((struct five_int) {10, 20, 30, 40, 50});

  return_two_double ((struct two_double) {10.0, 20.0});
  return_one_double_two_float ((struct one_double_two_float) {10.0, 20.0, 30.0});
  return_one_int_one_float_one_int ((struct one_int_one_float_one_int) {10, 20.0, 30});

  return_one_pointer ((struct one_pointer) {g_first_pointer});
  return_two_pointer ((struct two_pointer) {g_first_pointer, g_second_pointer});
  return_one_float_one_pointer ((struct one_float_one_pointer) {10.0, g_first_pointer});
  return_one_int_one_pointer ((struct one_int_one_pointer) {10, g_first_pointer});
  return_three_short_one_float ((struct three_short_one_float) {10, 20, 30, 40.0});

  return_one_int_one_double ((struct one_int_one_double) {10, 20.0});
  return_one_int_one_double_one_int ((struct one_int_one_double_one_int) {10, 20.0, 30});
  return_one_short_one_double_one_short ((struct one_short_one_double_one_short) {10, 20.0, 30});
  return_one_float_one_int_one_float ((struct one_float_one_int_one_float) {10.0, 20, 30.0});
  return_two_float ((struct two_float) { 10.0, 20.0});
  return_one_int_one_double_packed ((struct one_int_one_double_packed) {10, 20.0});
  return_one_int_one_long ((struct one_int_one_long) {10, 20});

  return_nested_one_float_three_base((struct nested_one_float_three_base) {
                                        10.0,
                                        (struct base_one_char) {
                                          'x'
                                        },
                                        (struct base_one_char) {
                                          'y'
                                        },
                                        (struct base_one_char) {
                                          'z'
                                        }
                                      });
  return_double_nested_one_float_one_nested((struct double_nested_one_float_one_nested) {
                                              10.0,
                                              (struct nested_one_float_three_base) {
                                                20.0,
                                                (struct base_one_char) {
                                                  'x'
                                                },
                                                (struct base_one_char) {
                                                  'y'
                                                },
                                                (struct base_one_char) {
                                                  'z'
                                                }
                                              }});
  return_nested_float_struct((struct nested_float_struct) {
                                10.0,
                                (struct base_float_struct) {
                                  20.0,
                                  30.0
                                }});
  return_six_double_three_int((struct six_double_three_int) {
                                10.0, 20.0, 1, 30.0, 40.0, 2, 50.0, 60.0, 3});

  return_base_class_one_char(base_class_one_char());
  return_nested_class_float_and_base(nested_class_float_and_base());
  return_double_nested_class_float_and_nested(double_nested_class_float_and_nested());
  return_base_class(base_class());
  // this is expected to fail
  return_sub_class(sub_class());
  return_derived_class(derived_class());

  return_vector_size_float32_8 (( vector_size_float32_8 ){1.5, 2.25});
  return_vector_size_float32_16 (( vector_size_float32_16 ){1.5, 2.25, 4.125, 8.0625});
  return_vector_size_float32_32 (( vector_size_float32_32 ){1.5, 2.25, 4.125, 8.0625, 7.89, 8.52, 6.31, 9.12});

  return_ext_vector_size_float32_2 ((ext_vector_size_float32_2){ 16.5, 32.25});
  return_ext_vector_size_float32_4 ((ext_vector_size_float32_4){ 16.5, 32.25, 64.125, 128.0625});
  return_ext_vector_size_float32_8 ((ext_vector_size_float32_8){ 16.5, 32.25, 64.125, 128.0625, 1.59, 3.57, 8.63, 9.12 });

  return 0; 
}
