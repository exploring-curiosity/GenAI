 CREATE TABLE user_chats (
    user_email VARCHAR(255) NOT NULL,
    smdg_terminal_code VARCHAR(10) NOT NULL,
    chat_history JSONB NOT NULL,
    PRIMARY KEY (user_email, smdg_terminal_code)
);
