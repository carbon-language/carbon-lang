enum TchkType {
  tchkNum, tchkString, tchkSCN, tchkNone
};

struct Operator {
  enum TchkType tchk[8];
};

struct Operator opTab[] = {
  {{tchkNum, tchkNum, tchkString} }
};

