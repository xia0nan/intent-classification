from requests import put, get

if __name__ == '__main__':
    question_list = ['this is a test',
                     'how can i open a 360 account?',
                     'wth is this?']

    # print(get('http://localhost:5000/', data={'question': 'this is a test'}).json())
    for question in question_list:
        print(get('http://localhost:5000/', data={'question': question}).json())