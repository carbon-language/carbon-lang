// RUN: clang -rewrite-test %s | clang

@interface foo @end
@interface GARF @end

int main()
{

@try  {
   MYTRY();
}

@catch (foo* localException) {
   MYCATCH();
   @throw;
}
}

