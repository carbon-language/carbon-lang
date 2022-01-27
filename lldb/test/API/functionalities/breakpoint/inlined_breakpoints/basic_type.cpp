// This file must have the following defined before it is included:
// T defined to the type to test (int, float, etc)
// T_CSTR a C string representation of the type T ("int", "float")
// T_VALUE_1 defined to a valid initializer value for TEST_TYPE (7 for int, 2.0 for float)
// T_VALUE_2, T_VALUE_3, T_VALUE_4 defined to a valid initializer value for TEST_TYPE that is different from TEST_VALUE_1
// T_PRINTF_FORMAT defined if T can be printed with printf
//
// An example for integers is below
#if 0

#define T int
#define T_CSTR "int"
#define T_VALUE_1 11001110
#define T_VALUE_2 22002220
#define T_VALUE_3 33003330
#define T_VALUE_4 44044440
#define T_PRINTF_FORMAT "%i"

#include "basic_type.cpp"

#endif

#include <cstdint>
#include <cstdio>

class a_class 
{
public:
    a_class (const T& a, const T& b) :
        m_a (a),
        m_b (b)
    {
    }

    ~a_class ()
    {
    }

    const T&
    get_a()
    {
        return m_a;
    } 

    void
    set_a (const T& a)
    {
        m_a = a;
    }

    const T&
    get_b()
    {
        return m_b;
    } 

    void
    set_b (const T& b)
    {
        m_b = b;
    }

protected:
    T m_a;
    T m_b;
};

typedef struct a_struct_tag {
    T a;
    T b;
} a_struct_t;


typedef union a_union_zero_tag {
    T a;
    double a_double;
} a_union_zero_t;

typedef struct a_union_nonzero_tag {
  double a_double;
  a_union_zero_t u;
} a_union_nonzero_t;


void Puts(char const *msg)
{
  std::puts(msg);
}

int 
main (int argc, char const *argv[])
{
    T a = T_VALUE_1;
    T* a_ptr = &a;
    T& a_ref = a;
    T a_array_bounded[2] = { T_VALUE_1, T_VALUE_2 };    
    T a_array_unbounded[] = { T_VALUE_1, T_VALUE_2 };

    a_class a_class_instance (T_VALUE_1, T_VALUE_2);
    a_class *a_class_ptr = &a_class_instance;
    a_class &a_class_ref = a_class_instance;

    a_struct_t a_struct = { T_VALUE_1, T_VALUE_2 };
    a_struct_t *a_struct_ptr = &a_struct;
    a_struct_t &a_struct_ref = a_struct;

    // Create a union with type T at offset zero
    a_union_zero_t a_union_zero;
    a_union_zero.a = T_VALUE_1;
    a_union_zero_t *a_union_zero_ptr = &a_union_zero;
    a_union_zero_t &a_union_zero_ref = a_union_zero;

    // Create a union with type T at a non-zero offset
    a_union_nonzero_t a_union_nonzero;
    a_union_nonzero.u.a = T_VALUE_1;
    a_union_nonzero_t *a_union_nonzero_ptr = &a_union_nonzero;
    a_union_nonzero_t &a_union_nonzero_ref = a_union_nonzero;

    a_struct_t a_struct_array_bounded[2]  = {{ T_VALUE_1, T_VALUE_2 }, { T_VALUE_3, T_VALUE_4 }};
    a_struct_t a_struct_array_unbounded[] = {{ T_VALUE_1, T_VALUE_2 }, { T_VALUE_3, T_VALUE_4 }};
    a_union_zero_t a_union_zero_array_bounded[2];
    a_union_zero_array_bounded[0].a = T_VALUE_1;
    a_union_zero_array_bounded[1].a = T_VALUE_2;
    a_union_zero_t a_union_zero_array_unbounded[] = {{ T_VALUE_1 }, { T_VALUE_2 }};
    
#ifdef T_PRINTF_FORMAT
    std::printf ("%s: a = '" T_PRINTF_FORMAT "'\n", T_CSTR, a);
    std::printf ("%s*: %p => *a_ptr = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_ptr, *a_ptr);
    std::printf ("%s&: @%p => a_ref = '" T_PRINTF_FORMAT "'\n", T_CSTR, &a_ref, a_ref);

    std::printf ("%s[2]: a_array_bounded[0] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_bounded[0]);
    std::printf ("%s[2]: a_array_bounded[1] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_bounded[1]);

    std::printf ("%s[]: a_array_unbounded[0] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_unbounded[0]);
    std::printf ("%s[]: a_array_unbounded[1] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_unbounded[1]);

    std::printf ("(a_class) a_class_instance.m_a = '" T_PRINTF_FORMAT "'\n", a_class_instance.get_a());
    std::printf ("(a_class) a_class_instance.m_b = '" T_PRINTF_FORMAT "'\n", a_class_instance.get_b());
    std::printf ("(a_class*) a_class_ptr = %p, a_class_ptr->m_a = '" T_PRINTF_FORMAT "'\n", a_class_ptr, a_class_ptr->get_a());
    std::printf ("(a_class*) a_class_ptr = %p, a_class_ptr->m_b = '" T_PRINTF_FORMAT "'\n", a_class_ptr, a_class_ptr->get_b());
    std::printf ("(a_class&) a_class_ref = %p, a_class_ref.m_a = '" T_PRINTF_FORMAT "'\n", &a_class_ref, a_class_ref.get_a());
    std::printf ("(a_class&) a_class_ref = %p, a_class_ref.m_b = '" T_PRINTF_FORMAT "'\n", &a_class_ref, a_class_ref.get_b());

    std::printf ("(a_struct_t) a_struct.a = '" T_PRINTF_FORMAT "'\n", a_struct.a);
    std::printf ("(a_struct_t) a_struct.b = '" T_PRINTF_FORMAT "'\n", a_struct.b);
    std::printf ("(a_struct_t*) a_struct_ptr = %p, a_struct_ptr->a = '" T_PRINTF_FORMAT "'\n", a_struct_ptr, a_struct_ptr->a);
    std::printf ("(a_struct_t*) a_struct_ptr = %p, a_struct_ptr->b = '" T_PRINTF_FORMAT "'\n", a_struct_ptr, a_struct_ptr->b);
    std::printf ("(a_struct_t&) a_struct_ref = %p, a_struct_ref.a = '" T_PRINTF_FORMAT "'\n", &a_struct_ref, a_struct_ref.a);
    std::printf ("(a_struct_t&) a_struct_ref = %p, a_struct_ref.b = '" T_PRINTF_FORMAT "'\n", &a_struct_ref, a_struct_ref.b);
    
    std::printf ("(a_union_zero_t) a_union_zero.a = '" T_PRINTF_FORMAT "'\n", a_union_zero.a);
    std::printf ("(a_union_zero_t*) a_union_zero_ptr = %p, a_union_zero_ptr->a = '" T_PRINTF_FORMAT "'\n", a_union_zero_ptr, a_union_zero_ptr->a);
    std::printf ("(a_union_zero_t&) a_union_zero_ref = %p, a_union_zero_ref.a = '" T_PRINTF_FORMAT "'\n", &a_union_zero_ref, a_union_zero_ref.a);

    std::printf ("(a_union_nonzero_t) a_union_nonzero.u.a = '" T_PRINTF_FORMAT "'\n", a_union_nonzero.u.a);
    std::printf ("(a_union_nonzero_t*) a_union_nonzero_ptr = %p, a_union_nonzero_ptr->u.a = '" T_PRINTF_FORMAT "'\n", a_union_nonzero_ptr, a_union_nonzero_ptr->u.a);
    std::printf ("(a_union_nonzero_t&) a_union_nonzero_ref = %p, a_union_nonzero_ref.u.a = '" T_PRINTF_FORMAT "'\n", &a_union_nonzero_ref, a_union_nonzero_ref.u.a);

    std::printf ("(a_struct_t[2]) a_struct_array_bounded[0].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[0].a);
    std::printf ("(a_struct_t[2]) a_struct_array_bounded[0].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[0].b);
    std::printf ("(a_struct_t[2]) a_struct_array_bounded[1].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[1].a);
    std::printf ("(a_struct_t[2]) a_struct_array_bounded[1].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[1].b);

    std::printf ("(a_struct_t[]) a_struct_array_unbounded[0].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[0].a);
    std::printf ("(a_struct_t[]) a_struct_array_unbounded[0].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[0].b);
    std::printf ("(a_struct_t[]) a_struct_array_unbounded[1].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[1].a);
    std::printf ("(a_struct_t[]) a_struct_array_unbounded[1].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[1].b);

    std::printf ("(a_union_zero_t[2]) a_union_zero_array_bounded[0].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_bounded[0].a);
    std::printf ("(a_union_zero_t[2]) a_union_zero_array_bounded[1].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_bounded[1].a);

    std::printf ("(a_union_zero_t[]) a_union_zero_array_unbounded[0].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_unbounded[0].a);
    std::printf ("(a_union_zero_t[]) a_union_zero_array_unbounded[1].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_unbounded[1].a);

#endif
    Puts("About to exit, break here to check values..."); // Set break point at this line.
    return 0;
}
