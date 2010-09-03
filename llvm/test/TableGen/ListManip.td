// RUN: tblgen %s
class Bli<string _t>
{
  string t = _t;
}

class Bla<list<Bli> _bli>
: Bli<!car(_bli).t>
{
}
