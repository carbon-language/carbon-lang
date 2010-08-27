// RUN: %clang %s -S -emit-llvm -o - | grep -e "define linkonce_odr.*_ZN6pr8007lsERNS_11std_ostreamERKNS_8StreamerINS_3FooEEE"
// XFAIL: *

namespace pr8007 {

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

} // namespace pr8007

int main(void)
{
    using namespace pr8007;
    Foo foo;
    cout << foo;
}
