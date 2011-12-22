int 
inner_sint (int value)
{
  return value;
}

int
outer_sint (int value)
{
  return inner_sint (value);
}

float
inner_float (float value)
{
  return value;
}

float 
outer_float (float value)
{
  return inner_float(value);
}

double
inner_double (double value)
{
  return value;
}

double 
outer_double (double value)
{
  return inner_double(value);
}

long double
inner_long_double (long double value)
{
  return value;
}

long double 
outer_long_double (long double value)
{
  return inner_long_double(value);
}

struct
large_return_struct
{
  long long first_long;
  long long second_long;
  long long third_long;
  long long fourth_long;

};

struct large_return_struct
return_large_struct (long long first, long long second, long long third, long long fourth)
{
  return (struct large_return_struct) {first, second, third, fourth};
}

int 
main ()
{
  int first_int = 123456;
  int second_int = 234567;

  outer_sint (first_int);
  outer_sint (second_int);

  float float_value = 12.34;
  
  outer_float (float_value);

  double double_value = -23.45;

  outer_double (double_value);

  long double long_double_value = -3456789.987654321;

  outer_long_double (long_double_value);

  return_large_struct (10, 20, 30, 40);

  return 0;
  
}
