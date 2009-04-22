
@protocol foo;

// Expressions
typedef typeof(@"foo" "bar") objc_string;
typedef typeof(@encode(int)) objc_encode;
typedef typeof(@protocol(foo)) objc_protocol;


// Types.
typedef typeof(id<foo>) objc_id_protocol_ty;

