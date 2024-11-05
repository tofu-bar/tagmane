public class Counter
{
    private int _count;
    private readonly object _lock = new object();

    public int Value
    {
        get
        {
            lock (_lock)
            {
                return _count;
            }
        }
    }

    public void Increment()
    {
        lock (_lock)
        {
            _count++;
        }
    }
    public void Decrement()
    {
        lock (_lock)
        {
            _count--;
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _count = 0;
        }
    }

    public int Count()
    {
        lock (_lock)
        {
            return _count;
        }
    }
}
