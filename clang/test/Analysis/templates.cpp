// RUN: %clang_cc1 -analyze -analyzer-checker=core -fblocks -verify %s

// Do not crash on this templated code which uses a block.
typedef void (^my_block)(void);
static void useBlock(my_block block){}
template<class T> class MyClass;
typedef MyClass<float> Mf;

template<class T>
class MyClass
{
public:
  MyClass() {}
  MyClass(T a);
  void I();
private:
 static const T one;
};

template<class T> const T MyClass<T>::one = static_cast<T>(1);
template<class T> inline MyClass<T>::MyClass(T a){}
template<class T> void MyClass<T>::I() {
  static MyClass<T>* mPtr = 0;
  useBlock(^{ mPtr = new MyClass<T> (MyClass<T>::one); });
};
int main(){
  Mf m;
  m.I();
}
