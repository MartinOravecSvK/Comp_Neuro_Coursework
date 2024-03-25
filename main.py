import os
import sys

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the new directory relative to the script's directory
new_directory = os.path.join(script_dir, 'src/')

# Change the current working directory to the new directory
sys.path.append(new_directory)

import q1
import q2
import q4
import q5

def main():
    print("Calculating question 1\n")
    q1.question1()
    print("\n\n")
    print("Calculating question 2\n")
    q2.question2()
    print("\n\n")
    print("Calculating question 4\n")
    q4.question4()
    print("\n\n")
    print("Calculating question 5\n")
    q5.question5()
    print("\n\n")


if __name__ == "__main__":
    main()