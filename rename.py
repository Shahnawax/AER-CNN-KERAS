'''
Created on Fri Nov 10 11:15:55 2017

@author: Muhammad Shahnawaz

This script is to rename all the files in Samples directory with numbers. The files by default have a long name so we replaced all the file names with numbers.
'''
import os
def rename_file(folder):
    i = 1
    for filename in os.listdir(folder):
        infilename = os.path.join(folder,filename)
        if not os.path.isfile(infilename):continue
        newname = os.path.join(folder,('{:d}.wav'.format(i)))
        i = i + 1
        os.rename(infilename,newname)
# code entry point
if __name__ == '__main__':
    folder = 'Samples'
    directs = os.listdir(folder)
    for directory in directs:
        rename_file(os.path.join(folder,directory))
