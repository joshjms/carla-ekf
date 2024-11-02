# Author: Joshua James (jjames001@e.ntu.edu.sg)
# EE3180 Design & Innovation Project AY2024/25 S1

from env import create_env

def main():
    env = create_env()

    try:
        env.run()
    except KeyboardInterrupt:
        env.display()
        print('Exiting...')
    except Exception as e:
        print('Error occurred: ', e)
        print('Exiting...')
    finally:
        print('Cleaning up...')
        env.cleanup()
        print('Done')

if __name__ == '__main__':
    main()