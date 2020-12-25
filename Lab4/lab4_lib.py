
def lookup(varname , x , content=False):
    
    try:
        print(varname , ':' , x.shape, end = ' ')
    except:
        print(varname , 'cannot show shape', end = ' ')
    if content:
        print(x)
    else:
        print()
