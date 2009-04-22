
@protocol foo;

typedef typeof(@"foo" "bar") objc_string;

typedef typeof(@encode(int)) objc_encode;


typedef typeof(@protocol(foo)) objc_protocol;

//const char *X;
