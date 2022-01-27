// RUN: %clang_cc1 %s -verify -fsyntax-only -Wvector-conversion -triple x86_64-apple-darwin10
typedef unsigned int v2u __attribute__ ((vector_size (8)));
typedef int v2s __attribute__ ((vector_size (8)));
typedef float v2f __attribute__ ((vector_size(8)));

void test1(v2u v2ua, v2s v2sa, v2f v2fa) {
  // Bitwise binary operators
  (void)(v2ua & v2ua);
  (void)(v2fa & v2fa); // expected-error{{invalid operands to binary expression}}

  // Unary operators
  (void)(~v2ua);
  (void)(~v2fa); // expected-error{{invalid argument type 'v2f' (vector of 2 'float' values) to unary}}

  // Comparison operators
  v2ua = (v2ua==v2sa); // expected-warning{{incompatible vector types assigning to 'v2u' (vector of 2 'unsigned int' values) from '__attribute__((__vector_size__(2 * sizeof(int)))) int' (vector of 2 'int' values}}
  v2sa = (v2ua==v2sa);

  // Arrays
  int array1[v2ua]; // expected-error{{size of array has non-integer type 'v2u' (vector of 2 'unsigned int' values}}
  int array2[17];
  // FIXME: error message below needs type!
  (void)(array2[v2ua]); // expected-error{{array subscript is not an integer}}

  v2u *v2u_ptr = 0;
  v2s *v2s_ptr;
  v2s_ptr = v2u_ptr; // expected-warning{{converts between pointers to integer types with different sign}}
}

void testLogicalVecVec(v2u v2ua, v2s v2sa, v2f v2fa) {
  // Logical operators
  v2ua = v2ua && v2ua; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2u'}}
  v2ua = v2ua || v2ua; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2u'}}

  v2ua = v2sa && v2ua; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2u' (vector of 2 'unsigned int' values)}}
  v2ua = v2sa || v2ua; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2u' (vector of 2 'unsigned int' values)}}

  v2ua = v2ua && v2fa; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2f' (vector of 2 'float' values)}}
  v2ua = v2ua || v2fa; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2f' (vector of 2 'float' values)}}

  v2ua = v2sa && v2fa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2f' (vector of 2 'float' values)}}
  v2ua = v2sa || v2fa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2f' (vector of 2 'float' values)}}

  v2sa = v2sa && v2sa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2s'}}
  v2sa = v2sa || v2sa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2s'}}

  v2sa = v2ua && v2ua; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2u'}}
  v2sa = v2ua || v2ua; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2u'}}

  v2sa = v2sa && v2ua; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2u' (vector of 2 'unsigned int' values)}}
  v2sa = v2sa || v2ua; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2u' (vector of 2 'unsigned int' values)}}

  v2sa = v2sa && v2fa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2f' (vector of 2 'float' values)}}
  v2sa = v2sa || v2fa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2f' (vector of 2 'float' values)}}

  v2sa = v2ua && v2fa; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2f' (vector of 2 'float' values)}}
  v2sa = v2ua || v2fa; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2f' (vector of 2 'float' values)}}

  v2fa = v2fa && v2fa; // expected-error {{logical expression with vector types 'v2f' (vector of 2 'float' values) and 'v2f'}}
  v2fa = v2fa || v2fa; // expected-error {{logical expression with vector types 'v2f' (vector of 2 'float' values) and 'v2f'}}

  v2fa = v2sa && v2fa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2f' (vector of 2 'float' values)}}
  v2fa = v2sa || v2fa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2f' (vector of 2 'float' values)}}

  v2fa = v2ua && v2fa; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2f' (vector of 2 'float' values)}}
  v2fa = v2ua || v2fa; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2f' (vector of 2 'float' values)}}

  v2fa = v2ua && v2ua; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2u'}}
  v2fa = v2ua || v2ua; // expected-error {{logical expression with vector types 'v2u' (vector of 2 'unsigned int' values) and 'v2u'}}

  v2fa = v2sa && v2sa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2s'}}
  v2fa = v2sa || v2sa; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2s'}}

  v2fa = v2sa && v2ua; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2u' (vector of 2 'unsigned int' values)}}
  v2fa = v2sa || v2ua; // expected-error {{logical expression with vector types 'v2s' (vector of 2 'int' values) and 'v2u' (vector of 2 'unsigned int' values)}}
}

void testLogicalVecScalar(v2u v2ua, v2s v2sa, v2f v2fa) {
  unsigned u1;
  v2ua = v2ua && u1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'unsigned int' is only supported in C++}} 
  v2ua = v2ua || u1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'unsigned int' is only supported in C++}} 

  v2sa = v2sa && u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2s' (vector of 2 'int' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2s' (vector of 2 'int' values) and 'unsigned int')}}
  v2sa = v2sa || u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2s' (vector of 2 'int' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2s' (vector of 2 'int' values) and 'unsigned int')}}

  v2ua = v2sa && u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2s' (vector of 2 'int' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2s' (vector of 2 'int' values) and 'unsigned int')}}
  v2ua = v2sa || u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2s' (vector of 2 'int' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2s' (vector of 2 'int' values) and 'unsigned int')}}
  v2sa = v2ua && u1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'unsigned int' is only supported in C++}}
  v2sa = v2ua || u1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'unsigned int' is only supported in C++}}

  v2ua = v2fa && u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'unsigned int')}}
  v2ua = v2fa || u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'unsigned int')}}

  v2sa = v2fa && u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'unsigned int')}}
  v2sa = v2fa || u1; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'unsigned int')}}

  int s1;
  v2ua = v2ua && s1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'int' is only supported in C++}}
  v2ua = v2ua || s1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'int' is only supported in C++}}

  v2sa = v2sa && s1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'int' is only supported in C++}}
  v2sa = v2sa || s1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'int' is only supported in C++}}

  v2ua = v2sa && s1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'int' is only supported in C++}}
  v2ua = v2sa || s1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'int' is only supported in C++}}

  v2sa = v2ua && s1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'int' is only supported in C++}}
  v2sa = v2ua || s1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'int' is only supported in C++}}

  v2ua = v2fa && s1; // expected-error {{cannot convert between scalar type 'int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'int'}}
  v2ua = v2fa || s1; // expected-error {{cannot convert between scalar type 'int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'int'}}

  v2sa = v2fa && s1; // expected-error {{cannot convert between scalar type 'int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'int'}}
  v2sa = v2fa || s1; // expected-error {{cannot convert between scalar type 'int' and vector type 'v2f' (vector of 2 'float' values) as implicit conversion would cause truncation}} expected-error {{invalid operands to binary expression ('v2f' (vector of 2 'float' values) and 'int'}}

  float f1;
  v2ua = v2ua && f1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'float' is only supported in C++}}
  v2ua = v2ua || f1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'float' is only supported in C++}}

  v2sa = v2sa && f1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'float' is only supported in C++}}
  v2sa = v2sa || f1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'float' is only supported in C++}}

  v2ua = v2sa && f1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'float' is only supported in C++}}
  v2ua = v2sa || f1; // expected-error {{logical expression with vector type 'v2s' (vector of 2 'int' values) and non-vector type 'float' is only supported in C++}}

  v2sa = v2ua && f1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'float' is only supported in C++}}
  v2sa = v2ua || f1; // expected-error {{logical expression with vector type 'v2u' (vector of 2 'unsigned int' values) and non-vector type 'float' is only supported in C++}}

  v2ua = v2fa && f1; // expected-error {{logical expression with vector type 'v2f' (vector of 2 'float' values) and non-vector type 'float' is only supported in C++}}
  v2ua = v2fa || f1; // expected-error {{logical expression with vector type 'v2f' (vector of 2 'float' values) and non-vector type 'float' is only supported in C++}}

  v2sa = v2fa && f1; // expected-error {{logical expression with vector type 'v2f' (vector of 2 'float' values) and non-vector type 'float' is only supported in C++}}
  v2sa = v2fa || f1; // expected-error {{logical expression with vector type 'v2f' (vector of 2 'float' values) and non-vector type 'float' is only supported in C++}}

}
