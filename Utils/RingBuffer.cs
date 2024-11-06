using System.Collections;
using System.Collections.Generic;
using System.Threading;

public class RingBuffer<T> : IEnumerable<T>
{
    private readonly T[] _buffer;
    private int _head;
    private int _tail;
    private int _count;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();

    public RingBuffer(int capacity)
    {
        _buffer = new T[capacity];
        _head = 0;
        _tail = 0;
        _count = 0;
    }

    public void Enqueue(T item)
    {
        _lock.EnterWriteLock();
        try
        {
            _buffer[_tail] = item;
            _tail = (_tail + 1) % _buffer.Length;

            if (_count == _buffer.Length)
            {
                _head = (_head + 1) % _buffer.Length;
            }
            else
            {
                _count++;
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public List<T> GetRecentItems()
    {
        _lock.EnterReadLock();
        try
        {
            T[] snapshot = new T[_count];
            for (int i = 0; i < _count; i++)
            {
                int index = (_head + i) % _buffer.Length;
                snapshot[i] = _buffer[index];
            }
            return new List<T>(snapshot);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public int Count
    {
        get
        {
            _lock.EnterReadLock();
            try
            {
                return _count;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }

    public int Capacity => _buffer.Length;
    
    public IEnumerator<T> GetEnumerator()
    {
        _lock.EnterReadLock();
        try
        {
            for (int i = 0; i < _count; i++)
            {
                int index = (_head + i) % _buffer.Length;
                yield return _buffer[index];
            }
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
