def any_true(list, predicate):
    for i in list:
        if predicate(i):
            return True
    return False

def any_false(list, predicate):
    return any_true(list, lambda x: not predicate(x))

def all_true(list, predicate):
    return not any_false(list, predicate)

def all_false(list, predicate):
    return not any_true(list, predicate)

def prependLines(prependStr, str):
    return ('\n'+prependStr).join(str.splitlines())

def pprint(object, useRepr=True):
    def recur(ob):
        return pprint(ob, useRepr)
    def wrapString(prefix, string, suffix):
        return '%s%s%s' % (prefix, 
                           prependLines(' ' * len(prefix),
                                        string),
                           suffix)
    def pprintArgs(name, args):
        return wrapString(name + '(', ',\n'.join(map(recur,args)), ')')
                            
    if isinstance(object, tuple):
        return wrapString('(', ',\n'.join(map(recur,object)), 
                          [')',',)'][len(object) == 1])
    elif isinstance(object, list):
        return wrapString('[', ',\n'.join(map(recur,object)), ']')
    elif isinstance(object, set):
        return pprintArgs('set', list(object))
    elif isinstance(object, dict):
        elts = []
        for k,v in object.items():
            kr = recur(k)
            vr = recur(v)
            elts.append('%s : %s' % (kr, 
                                     prependLines(' ' * (3 + len(kr.splitlines()[-1])),
                                                  vr)))
        return wrapString('{', ',\n'.join(elts), '}')
    else:
        if useRepr:
            return repr(object)
        return str(object)

def prefixAndPPrint(prefix, object, useRepr=True):
    return prefix + prependLines(' '*len(prefix), pprint(object, useRepr))
