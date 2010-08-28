// RUN: %clang %s -S -emit-llvm -o - | grep -e "define linkonce_odr.*_ZlsR11std_ostreamRK8StreamerI3FooE"
// PR8007: friend function either not instantiated.

struct std_ostream
{
  int dummy;
};

std_ostream cout;

template <typename STRUCT_TYPE>
struct Streamer
{
    friend std_ostream& operator << (std_ostream& o, const Streamer& f)
        {
            Streamer s(f);
            s(o);
            return o;
        }

    Streamer(const STRUCT_TYPE& s) : s(s) {}

    const STRUCT_TYPE& s;
    void operator () (std_ostream&) const;
};

typedef struct Foo {} Foo;

std_ostream& operator << (std_ostream& o, const Streamer<Foo>& f);
/*std_ostream& operator << (std_ostream& o, const Streamer<Foo>& f)
{
  // Sema should flag this as a redefinition
}*/

template <>
void Streamer<Foo>::operator () (std_ostream& o) const
{
}

int main(void)
{
    Foo foo;
    cout << foo;
}
