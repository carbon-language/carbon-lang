int main() {
    int i = 0;
    for (int j = 3; j < 20; j++)
    {
        i += j;
        i = i - 1; // break here
    }
    return i;
}
