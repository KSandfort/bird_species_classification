import os

if __name__ == '__main__':
    print('PyCharm')
    # Creates a new file
    with open('myfile.txt', 'w') as fp:
        fp.write("New file created")