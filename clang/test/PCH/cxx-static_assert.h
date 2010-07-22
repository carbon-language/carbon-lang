// Header for PCH test cxx-static_assert.cpp





template<int N> struct T {
    static_assert(N == 2, "N is not 2!");
};
