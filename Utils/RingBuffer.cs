using System.Collections;
using System.Collections.Generic;
using System.Threading;

public class RingBuffer<T> : IEnumerable<T>
{
    private readonly T[] _buffer;
    private int _head;
    private int _tail;
    private int _count;
    private int _lastReadPosition = -1; // 最後に読み取った位置を記録
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
    
    /// <summary>
    /// 前回読み取り以降に追加された新しいアイテムのみを取得
    /// </summary>
    public List<T> GetNewItems()
    {
        _lock.EnterWriteLock(); // WriteLockを使用して_lastReadPositionの更新も保護
        try
        {
            if (_lastReadPosition == -1)
            {
                // 初回読み取りの場合は全てのアイテムを返す
                var allItems = GetRecentItemsInternal();
                _lastReadPosition = _tail; // 現在のtail位置を記録
                return allItems;
            }
            
            List<T> newItems = new List<T>();
            
            // 前回読み取り位置から現在のtail位置まで
            int currentLastRead = _lastReadPosition;
            int currentTail = _tail;
            
            if (currentLastRead != currentTail)
            {
                int start = currentLastRead;
                int end = currentTail;
                
                if (start < end)
                {
                    // 通常のケース
                    for (int i = start; i < end; i++)
                    {
                        newItems.Add(_buffer[i % _buffer.Length]);
                    }
                }
                else
                {
                    // リングバッファがラップアラウンドした場合
                    for (int i = start; i < _buffer.Length; i++)
                    {
                        newItems.Add(_buffer[i]);
                    }
                    for (int i = 0; i < end; i++)
                    {
                        newItems.Add(_buffer[i]);
                    }
                }
            }
            
            _lastReadPosition = currentTail; // 読み取り位置を更新
            return newItems;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
    
    /// <summary>
    /// 内部用のGetRecentItemsメソッド（ロックを取得済みの前提）
    /// </summary>
    private List<T> GetRecentItemsInternal()
    {
        T[] snapshot = new T[_count];
        for (int i = 0; i < _count; i++)
        {
            int index = (_head + i) % _buffer.Length;
            snapshot[i] = _buffer[index];
        }
        return new List<T>(snapshot);
    }
    
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
