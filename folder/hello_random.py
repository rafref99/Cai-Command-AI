import random

def generate_random_digits(length):
    # Generates a string of 'length' random digits (0-9)
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])

if __name__ == "__main__":
    random_number = generate_random_digits(100)
    print("hello" + random_number)