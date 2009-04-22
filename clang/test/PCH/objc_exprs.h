
@protocol foo;
@class itf;

// Expressions
typedef typeof(@"foo" "bar") objc_string;
typedef typeof(@encode(int)) objc_encode;
typedef typeof(@protocol(foo)) objc_protocol;


// Types.
typedef typeof(id<foo>) objc_id_protocol_ty;

typedef typeof(itf*) objc_interface_ty;
typedef typeof(itf<foo>*) objc_qual_interface_ty;
