// RUN: clang -fsyntax-only %s

@protocol MyProtocol
@end

id<MyProtocol> obj_p = 0;

int main()
{
  obj_p = 0;
}

