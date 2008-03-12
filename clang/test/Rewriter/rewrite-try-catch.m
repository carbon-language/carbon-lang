// RUN: clang -rewrite-test %s 

@interface Foo @end
@interface GARF @end

void foo() {
  @try  { TRY(); } 
  @catch (...) { SPLATCH(); @throw; }
}

int main()
{

@try  {
   MYTRY();
}

@catch (Foo* localException) {
   MYCATCH();
   @throw;
}
}

