// RUN: clang-cc -E %s | grep '^"x ## y";$'
#define hash_hash # ## # 
#define mkstr(a) # a 
#define in_between(a) mkstr(a) 
#define join(c, d) in_between(c hash_hash d) 
join(x, y);

