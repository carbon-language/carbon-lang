class S
{
public:
  S () { }
  S (S &obj);

  S operator+ (const S &s);

  int a;
};

S::S (S &obj)
{
  a = obj.a;
}

S
S::operator+ (const S &s)
{
  S res;

  res.a = a + s.a;

  return res;
}

S
f (int i)
{
  S s;

  s.a = i;

  return s;
}

int
g (const S &s)
{
  return s.a;
}

class A
{
public:
  A operator+ (const A &);
  int a;
};

A
A::operator+ (const A &obj)
{
  A n;

  n.a = a + obj.a;

  return n;
}

A
p ()
{
  A a;
  a.a = 12345678;
  return a;
}

A
r ()
{
  A a;
  a.a = 10000000;
  return a;
}

A
q (const A &a)
{
  return a;
}

class B
{
public:
  int b[1024];
};

B
makeb ()
{
  B b;
  int i;

  for (i = 0; i < 1024; i++)
    b.b[i] = i;

  return b;
}

int
getb (const B &b, int i)
{
  return b.b[i];
}

class C
{
public:
  C ();
  ~C ();

  A operator* ();

  A *a_ptr;
};

C::C ()
{
  a_ptr = new A;
  a_ptr->a = 5678;
}

C::~C ()
{
  delete a_ptr;
}

A
C::operator* ()
{
  return *a_ptr;
}

#define TYPE_INDEX 1

enum type
{
  INT,
  CHAR
};

union U
{
public:
  U (type t);
  type get_type ();

  int a;
  char c;
  type tp[2];
};

U::U (type t)
{
  tp[TYPE_INDEX] = t;
}

U
make_int ()
{
  return U (INT);
}

U
make_char ()
{
  return U (CHAR);
}

type
U::get_type ()
{
  return tp[TYPE_INDEX];
}

int
main ()
{
  int i = g(f(0));
  A a = q(p() + r());

  B b = makeb ();
  C c;

  return i + getb(b, 0);  /* Break here */
}
