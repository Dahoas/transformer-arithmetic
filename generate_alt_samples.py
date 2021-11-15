with open('alt_train.txt', 'w') as f:
    for i in range(1,2000):
        f.write('x*(x-%d)=x*x-%d*x;\n' % (i,i))