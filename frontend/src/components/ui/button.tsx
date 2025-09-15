import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary-glow hover:shadow-[0_0_20px_hsl(var(--primary-glow)/0.4)] hover:scale-105",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 hover:shadow-[0_0_20px_hsl(var(--destructive)/0.4)]",
        outline: "border border-border bg-card/50 backdrop-blur-sm text-foreground hover:bg-primary/10 hover:border-primary/50 hover:text-primary",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/90 hover:shadow-[0_0_20px_hsl(var(--secondary)/0.4)]",
        accent: "bg-gradient-to-r from-accent to-primary text-accent-foreground hover:shadow-[0_0_30px_hsl(var(--accent)/0.5)] hover:scale-105",
        ghost: "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
        link: "text-primary underline-offset-4 hover:underline hover:text-primary-glow",
        hero: "bg-gradient-to-r from-primary via-secondary to-accent text-primary-foreground font-semibold hover:shadow-[0_0_40px_hsl(var(--primary)/0.6)] hover:scale-105 border-0",
        glass: "bg-white/5 backdrop-blur-md border border-white/10 text-foreground hover:bg-white/10 hover:border-primary/30",
        success: "bg-success text-success-foreground hover:bg-success/90 hover:shadow-[0_0_20px_hsl(var(--success)/0.4)]",
        warning: "bg-warning text-warning-foreground hover:bg-warning/90 hover:shadow-[0_0_20px_hsl(var(--warning)/0.4)]",
      },
      size: {
        default: "h-11 px-6 py-2",
        sm: "h-9 rounded-md px-4 text-xs",
        lg: "h-12 rounded-lg px-8 text-base",
        xl: "h-14 rounded-xl px-10 text-lg",
        icon: "h-11 w-11",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
