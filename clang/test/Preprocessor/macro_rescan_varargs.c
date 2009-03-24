// RUN: clang-cc -E %s | grep -F "1: F, (, 'a', 'b', );" &&
// RUN: clang-cc -E %s | grep -F "2: 'a' + 'b';"
#define LPAREN ( 
#define RPAREN ) 
#define F(x, y) x + y 
#define ELLIP_FUNC(...) __VA_ARGS__ 

1: ELLIP_FUNC(F, LPAREN, 'a', 'b', RPAREN); /* 1st invocation */ 
2: ELLIP_FUNC(F LPAREN 'a', 'b' RPAREN); /* 2nd invocation */ 

