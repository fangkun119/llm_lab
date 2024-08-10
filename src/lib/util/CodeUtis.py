import inspect


def print_method(method, print_markdown: bool = True):
    source_code = inspect.getsource(method)
    if print_markdown:
        print(f"""```python\n
        {source_code}\n
        ```\n""")
    else:
        print(source_code)
