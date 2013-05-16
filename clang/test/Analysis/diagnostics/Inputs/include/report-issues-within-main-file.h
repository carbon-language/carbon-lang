template<typename _Tp>
class auto_ptr {
private:
  _Tp* _M_ptr;
public: 
  auto_ptr(_Tp* __p = 0) throw() : _M_ptr(__p) { }
  ~auto_ptr() { delete _M_ptr; }
};

void cause_div_by_zero_in_header(int in) {
  int h = 0;
  h = in/h;
  h++;
}

void do_something (int in) {
  in++;
  in++;
}