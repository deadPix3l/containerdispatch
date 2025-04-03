# Container Dispatch


### why? 
Because this code doesnt work natively:
```python
@dispatch.register
def _(arg: list[int|str]):
    return "this is a list of ints and strings!"
```

the stdlib `functools.singledispatch` returns:
```python
TypeError: Invalid annotation for 'arg'. list[int|str] is not a class.
```

This means we have to simplify the annotation at the expense of complicating the function, and less support for mypy:
```python
@dispatch.register
def _(arg: list):
    # list[int|str]
    if all(isinstance(i, (int,str)) for i in arg):
        return "this is a list of ints and strings!"

    # elif <other types>:
        # other logic

    else:
        # list[any]
```

Personally I find this antitheitcal to singledispatch, which is supposed to solve this exact problem!
The entire reason I reached for singledispatch was to replace these if-ridden router functions, and I
refused to accept this.
