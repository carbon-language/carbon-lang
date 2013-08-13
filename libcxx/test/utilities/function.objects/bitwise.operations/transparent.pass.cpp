#include <functional>
#include <string>

template <class _Tp>
struct is_transparent
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Up> static __two __test(...);
    template <class _Up> static char __test(typename _Up::is_transparent* = 0);
public:
    static const bool value = sizeof(__test<_Tp>(0)) == 1;
};


int main () {
#if _LIBCPP_STD_VER > 11

    static_assert ( !is_transparent<std::bit_and<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_and<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_and<void>>::value, "" );
    static_assert (  is_transparent<std::bit_and<>>::value, "" );

    static_assert ( !is_transparent<std::bit_or<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_or<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_or<void>>::value, "" );
    static_assert (  is_transparent<std::bit_or<>>::value, "" );

    static_assert ( !is_transparent<std::bit_xor<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_xor<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_xor<void>>::value, "" );
    static_assert (  is_transparent<std::bit_xor<>>::value, "" );

    static_assert ( !is_transparent<std::bit_not<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_not<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_not<void>>::value, "" );
    static_assert (  is_transparent<std::bit_not<>>::value, "" );
    
#endif

    return 0;
    }
