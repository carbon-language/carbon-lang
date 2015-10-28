class Foo
{
public:
    int Bar(int x, int y)
    {
        return x + y;
    }
};

int main()
{
    Foo f;
    f.Bar(1, 2);
}
