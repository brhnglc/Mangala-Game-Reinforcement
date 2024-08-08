import gym
from gym import spaces
import random
import pygame
import sys

#PyGame screen sizes
WIDTH,HEIGHT = 720, 236



class MangalaEnv(gym.Env):

    def convert_move(self, move, player):
        """
        Girdilerin pocket karşılıklarına çevrilmesi
        """
        if player == 1:
            return move
        if player == 2:
            return move+7
        return False

    def valid_move(self, pocket_position, player):
        """
        Hamle yapılıcak taş var mı diye kontrol ediliyor
        """
        player_1_side = (0 <= pocket_position <= 5)
        player_2_side = (7 <= pocket_position <= 12)

        if self.pockets[pocket_position] > 0:
            if player_1_side and player==1:
                return True
            if player_2_side and player==2:
                return True

        return

    def initialize_board(self):
        """
        Tahtanın doğru degerlerde başlatılması
        """
        num_stones_on_start = 4
        pockets = [num_stones_on_start]*14
        pockets[6] = 0
        pockets[13] = 0

        return pockets

    def check_game_over(self):
        """ Bir tarafın kuyusunda taş kalmaması durumunda oyunun bitmesi ve karşı
        tarafta taş kalması durumunda taşlarını bitiren oyuncunun hazinesinde toplanması
        """

        game_over = False

        empty_player_1 = sum(self.pockets[:6]) == 0
        empty_player_2 = sum(self.pockets[7:13]) == 0

        if empty_player_2:
            self.pockets[13] += sum(self.pockets[:6])
            self.pockets[:6] = [0]*6
            game_over = True

        if empty_player_1:
            self.pockets[6] += sum(self.pockets[7:13])
            self.pockets[7:13] = [0]*6
            game_over = True

        return game_over

    def determine_winner(self):
        """ Hazine durumlarına göre kazananın belirlenmesi
        """
        if self.pockets[13]>self.pockets[6]:
            return "Player 2"
        elif self.pockets[13]<self.pockets[6]:
            return "Player 1"
        return "Draw"

    def switch_player(self, player):
        """
        Tur geçişlerinde oyuncu geçişi
        """

        if player == 1:
            return 2
        return 1

    def capture(self, pocket_position, mangala_pocket,turn):
        """ pocket position : oynanan hamle
            mangala pocket : aktif oyuncunun hazinesin indexi

            eger oynanan son taş kendi boş kuyusuna denk geliyorsa
            ve de kuyunun karşıt kuyusu doluysa hem kendi kuyusundaki hemde
            karşıt kuyudaki taşları hazinesine eklenmesi
        """
        opposite_pocket_dict = {0: 12, 1:11, 2:10, 3:9, 4:8, 5:7,
                                7:5, 8:4, 9:3, 10:2, 11:1, 12:0}
        if(self.pockets[opposite_pocket_dict[pocket_position]] != 0):
            opposite_pocket = opposite_pocket_dict[pocket_position]

            self.pockets[mangala_pocket] += self.pockets[pocket_position]
            self.pockets[pocket_position] = 0


            self.pockets[mangala_pocket] += self.pockets[opposite_pocket]
            self.pockets[opposite_pocket] = 0


        return True

    def capture_even(self,pocket_position,mangala_pocket,turn):
        """
        Eger oyuncunun son taşı rakibin kuyusundaki taşların toplamını çift yapıyorsa
        o kuyudaki tüm taşları hazinesine katar.
        """

        self.pockets[mangala_pocket] += self.pockets[pocket_position]
        self.pockets[pocket_position] = 0

        return True

    def simulate_move(self, pocket_position, player):
        """
        Seçilen hamlenin oynanması
        """
        capture_amnt_p1 = 0
        capture_amnt_p2 = 0
        go_again_amnt = 0
        pockets = self.pockets
        #Geriye bir taş bırakma ve eger hazinede tek taş varsa sağa taşıma
        stones_drawn = pockets[pocket_position] #eldeki taş sayısı
        if(stones_drawn == 1):
            pockets[pocket_position] = 0
        elif(stones_drawn != 0):
            pockets[pocket_position] = 1
            stones_drawn -= 1

        # Hamle yapılan kuynun sağa doğru birer birer bırakılarak ilerlenmesi
        while stones_drawn > 0:
            pocket_position += 1

            if pocket_position > len(pockets)-1:
                pocket_position = 0


            #Karşıdakinin kuyusuna gelinmesi durumunda bir sonraki kuyuya geçme
            mangala_1_position = pocket_position==6
            mangala_2_position = pocket_position==13
            player_1 = player == 1
            player_2 = player == 2
            player1_capture_true = False
            player2_capture_true = False
            if mangala_1_position and player_2:
                continue
            if mangala_2_position and player_1:
                continue

            # Taşları bırakma
            pockets[pocket_position] += 1
            stones_drawn -= 1

        #çift sayı yaptı mı
        end_with_even = pockets[pocket_position] % 2 == 0

        # Son hamlenin kendi kuyusunda bitirilmesi
        end_on_player_1_side = (0 <= pocket_position <= 5)
        end_on_player_2_side = (7 <= pocket_position <= 12)

        #son taşa gelinmesi durumu kontrolü
        stone_was_empty = pockets[pocket_position] == 1

        #boş kuyuya gelinmesi durumunda capture
        # Player 1 capture
        if player_1 and end_on_player_1_side and stone_was_empty:
            player1_capture_true = self.capture(pocket_position, 6,"1")

        # Player 2 capture
        if player_2 and end_on_player_2_side and stone_was_empty:
            player2_capture_true = self.capture(pocket_position, 13,"2")
            
            
        if player1_capture_true:
            capture_amnt_p1 +=1
        if player2_capture_true:
            capture_amnt_p2 +=1

        #çift yapma durumuna göre captur
        if player_1 and end_on_player_2_side and end_with_even:
            player1_capture_true =self.capture_even(pocket_position,6,"1")
        if player_2 and end_on_player_1_side and end_with_even:
            player2_capture_true = self.capture_even(pocket_position,13,"2")
            
        if(player1_capture_true):
            capture_amnt_p1 +=1
        if(player2_capture_true):
            capture_amnt_p2 +=1


        # son taşın kendi kuyusuna gelme durumu göz önünde bulundurarak hamlenin kime gecicegi
        if mangala_1_position and player_1:
            next_player = player
            go_again_amnt +=1
        elif mangala_2_position and player_2:
            next_player = player
        else:
            next_player = self.switch_player(player)

        game_over = self.check_game_over()

        return next_player, game_over ,capture_amnt_p1,capture_amnt_p2,go_again_amnt


    def turn_table(self,pockets):
        return pockets[7:]+pockets[:7]
    
    def show_end_game_alert(self,winner):
        """
        Render mode human iken oyunun bitmesi durumunda kimin kazandıgını gösteren ekran ve sistemin kapatılması
        """
        if(self.render_mode == "human"):
            pygame.quit()  
            pygame.init()
            
            screen = pygame.display.set_mode((400, 100))
            pygame.display.set_caption("Mangala Oyunu Sonu")
            screen.fill((255, 253, 208))
            font = pygame.font.Font(None, 30)
            text = font.render(f"Oyuncu1:{self.pockets[6]}", True, "black")
            screen.blit(text, (30, 30))
            text = font.render(f"Oyuncu2:{self.pockets[13]}", True, "black")
            screen.blit(text, (160, 30))
            text = font.render(f"Oyuncu {winner} Kazandı!", True, "black")
            screen.blit(text, (30, 60))
            pygame.display.flip()    
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

    


    #GYM ENV

    def __init__(self):
        super(MangalaEnv, self).__init__()
        self.background_image = pygame.image.load("mangala_board.jpg")
        self.prev_mov1 = -1
        self.prev_mov2 = -1
        self.render_mode = "train"
        # Define action and observation space
        self.action_space = spaces.Discrete(6)  # 6 possible pockets
        self.observation_space = spaces.Box(low=0, high=48, shape=(14,), dtype=int)

        # Initialize the board
        self.pockets = self.initialize_board()
        self.player_turn = random.randint(1, 2)
        self.game_over = False
        
        self.capture_amount = 0
        self.capture_amount_p1 = 0
        self.capture_amount_p2 = 0
    def reset(self):
        # Reset the board

        self.pockets = self.initialize_board()
        self.player_turn = random.randint(1, 2)
        self.game_over = False
        self.capture_amount_p1 = 0
        self.capture_amount_p2 = 0
        self.go_again_amount = 0
        
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Mangala")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.render()



        return self.get_observation()

    def step(self, action):
        reward = 0
        move = self.convert_move(action, self.player_turn)
        if not self.valid_move(move, self.player_turn):
            assert("invalid_move")
            return self.get_observation(), 0, False, {}


        if(self.player_turn == 1):
            self.prev_mov1 = move
        if(self.player_turn == 2):
            self.prev_mov2 = move-7

        next_player, game_over,capture_amnt_p1,capture_amnt_p2,go_again_amnt = self.simulate_move(move, self.player_turn)
        
        
        self.capture_amount_p1 += capture_amnt_p1
        self.capture_amount_p2 += capture_amnt_p2
        self.go_again_amount += go_again_amnt
        
        self.player_turn = next_player
        self.game_over = game_over

        if game_over:
            winner = self.determine_winner()
            if winner == "Player 1":
                self.show_end_game_alert("1")
                reward += 1
                #print("Player 1 Win"+"n"+"n"*5)
            elif winner == "Player 2":
                self.show_end_game_alert("2")
                reward += -1
                #print("Player 2 Win"+"n"+"n"*5)

        if self.render_mode == 'human':
            self.render(move=move)

        return self.get_observation(), reward, game_over, {}

    def render(self,move="none"):
        if(self.render_mode == "human"):
            #print("prev1->",self.prev_mov1,"prev2->",self.prev_mov2)
                     # array to pocket res move 
            upper_pockets = self.pockets[:6]
            lower_pockets = self.pockets[7:-1]
    
            upper_pockets_marker = [1, 2, 3, 4, 5, 6]
            lower_pockets_marker = [1, 2, 3, 4, 5, 6]
            mangala_1 = self.pockets[-1]
            mangala_2 = self.pockets[6]
            self.screen.fill("white")
            # Oyuncu turn yazısı
            turn_text = self.font.render("Player {}'s Turn".format(self.player_turn), True, "black")
            self.screen.blit(turn_text, (10, 10))
            
            self.screen.blit(self.background_image, (0, 0))
            # Tahta çizimi
            #pygame.draw.rect(self.screen, "black", (50, 50, 700, 500), 2)
            # Pockets çizimi
            
            color1,color2 = "black","black"
            for i in range(6):
                if(i == self.prev_mov1):
                    color1 = "blue"
                else:
                    color1 = "black"
                if(i == self.prev_mov2):
                    color2 = "red"
                else:
                    color2 = "black"
                if(i>=3):
                    pad = 40
                else:
                    pad = 0
                    
                #pygame.draw.rect(self.screen, "black", (100 + i * 100, 100, 100, 100), 2)
                #pygame.draw.rect(self.screen, "black", (100 + i * 100, 300, 100, 100), 2)
                #pygame.draw.rect(self.screen, "black", (100 + i * 100, 100, 100, 200), 2)
                font2 = pygame.font.Font(None, 28)
                upper_stones_marker = font2.render(str(upper_pockets_marker[i]), True, "brown")
                upper_stones_marker.set_alpha(128)
                lower_stones_marker = font2.render(str(lower_pockets_marker[i]), True, "brown")
                lower_stones_marker.set_alpha(160)
                self.screen.blit(upper_stones_marker, (575-pad - i * 80, 12))
                self.screen.blit(lower_stones_marker, (135+pad + i * 80, 200))
                upper_stones = self.font.render(str(upper_pockets[i]), True, color1)
                lower_stones = self.font.render(str(lower_pockets[i]), True, color2)
                self.screen.blit(upper_stones, (575-pad - i * 80, 55))
                self.screen.blit(lower_stones, (135+pad + i * 80, 150))
            # Mangala çizimi
            mangala_1_text = self.font.render(str(mangala_1), True, "black")
            mangala_2_text = self.font.render(str(mangala_2), True, "black")
            if(int(mangala_1) >= 10):
                pad_on1 = 5
            else:
                pad_on1 = 0
                
            if(int(mangala_2) >= 10):
                pad_on2 = 5
            else:
                pad_on2 = 0
                
            self.screen.blit(mangala_1_text, (653-pad_on1, 100))
            self.screen.blit(mangala_2_text, (58-pad_on2, 100))
            pygame.display.flip()
            

    def close(self):
        if(self.render_mode == "human"):
             pygame.quit()
             sys.exit()

    def get_observation(self):
        return self.pockets

